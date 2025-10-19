"""Pipeline for collecting cat videos and extracting audio and frames."""

from __future__ import annotations
import concurrent.futures
import csv
import json
import logging
import os
import pathlib
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence
from uuid import uuid4

try:  # Optional dependency - handled gracefully later
    import cv2
except Exception as exc:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:  # Optional dependency - handled gracefully later
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:  # Optional dependency - handled gracefully later
    from moviepy.editor import VideoFileClip  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    VideoFileClip = None  # type: ignore
    MOVIEPY_IMPORT_ERROR = exc
else:
    MOVIEPY_IMPORT_ERROR = None

try:  # Optional dependency - handled gracefully later
    from pydub import AudioSegment  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    AudioSegment = None  # type: ignore
    PYDUB_IMPORT_ERROR = exc
else:
    PYDUB_IMPORT_ERROR = None

try:  # Optional dependency - handled gracefully later
    from PIL import Image  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None
try:  # Optional dependency - handled gracefully later
    from ultralytics import YOLO  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    YOLO = None  # type: ignore
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None

try:  # Optional dependency - handled gracefully later
    import yt_dlp  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    yt_dlp = None  # type: ignore
    YTDLP_IMPORT_ERROR = exc
else:
    YTDLP_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("alim_pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ALIM")


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------
RUN_ID = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{uuid4().hex[:6]}"
logger.info("🎬 Lancement pipeline ALIM | run_id=%s", RUN_ID)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineConfig:
    search_queries: Sequence[str]
    max_results_per_query: int
    max_videos_per_run: int
    min_duration: float
    max_duration: float
    yolo_model: str
    yolo_conf_threshold: float
    n_frames_per_video: int
    diversity_threshold: int
    max_tries_factor: int
    dataset_json: pathlib.Path
    whitelist_file: pathlib.Path
    min_disk_gb_extract: float = 0.3
    min_disk_gb_yolo: float = 0.3
    search_retries: int = 2
    search_retry_backoff: float = 2.5
    download_retries: int = 3
    download_retry_backoff: float = 3.5
    ydl_socket_timeout: float = 20.0


CONFIG = PipelineConfig(
    search_queries=[
        "cat meowing",
        "kitten meow",
        "chat qui miaule",
        "chat mignon",
        "chat drôle",
        "cat funny",
        "cat talking",
        "kitten playing",
        "chat qui parle",
    ],
    max_results_per_query=10,
    max_videos_per_run=20,
    min_duration=5,
    max_duration=15,
    yolo_model="yolov8n.pt",
    yolo_conf_threshold=0.60,
    n_frames_per_video=5,
    diversity_threshold=8,
    max_tries_factor=5,
    dataset_json=pathlib.Path("alim_dataset.json"),
    whitelist_file=pathlib.Path("data") / "whitelist_urls.txt",
)

if MOVIEPY_IMPORT_ERROR:
    logger.warning(
        "moviepy indisponible lors du démarrage : %s | extraction audio désactivée",
        MOVIEPY_IMPORT_ERROR,
    )
if PYDUB_IMPORT_ERROR:
    logger.warning(
        "PyDub indisponible lors du démarrage : %s | fallback MP3->WAV désactivé",
        PYDUB_IMPORT_ERROR,
    )
if PIL_IMPORT_ERROR:
    logger.warning(
        "Pillow (PIL) indisponible lors du démarrage : %s | captioning désactivé",
        PIL_IMPORT_ERROR,
    )
if YTDLP_IMPORT_ERROR:
    logger.error(
        "yt_dlp indisponible lors du démarrage : %s | téléchargement YouTube impossible",
        YTDLP_IMPORT_ERROR,
    )
if NUMPY_IMPORT_ERROR:
    logger.error(
        "NumPy indisponible lors du démarrage : %s | opérations sur les images impossibles",
        NUMPY_IMPORT_ERROR,
    )
if CV2_IMPORT_ERROR:
    logger.error(
        "OpenCV (cv2) indisponible lors du démarrage : %s | extraction de frames impossible",
        CV2_IMPORT_ERROR,
    )
if YOLO_IMPORT_ERROR:
    logger.warning(
        "Ultralytics YOLO indisponible lors du démarrage : %s | détection de chats désactivée",
        YOLO_IMPORT_ERROR,
    )

NUMPY_AVAILABLE = np is not None
CV2_AVAILABLE = cv2 is not None and NUMPY_AVAILABLE
YTDLP_AVAILABLE = yt_dlp is not None
YOLO_AVAILABLE = YOLO is not None


# ---------------------------------------------------------------------------
# Optional captioning dependencies
# ---------------------------------------------------------------------------
CAPTIONING_ENABLED = False
device = "cpu"
processor = None
blip_model = None

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore
    import torch  # type: ignore
    CAPTIONING_ENABLED = True
except Exception as exc:  # pragma: no cover - optional dependency
    logger.warning("BLIP indisponible (transformers/torch/PIL): %s", exc)
    CAPTIONING_ENABLED = False

if CAPTIONING_ENABLED and Image is None:
    logger.warning(
        "Pillow est requis pour le captioning BLIP. Veuillez installer 'Pillow'."
    )
    CAPTIONING_ENABLED = False


# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
BASE = pathlib.Path(".")
RAW_DIR = BASE / "data" / "raw_videos"
FRAME_DIR = BASE / "data" / "frames"
AUDIO_DIR = BASE / "data" / "audio"
ACCEPTED_DIR = BASE / "data" / "accepted_videos"
MANIFEST_DIR = BASE / "data" / "manifest"

for directory in (RAW_DIR, FRAME_DIR, AUDIO_DIR, ACCEPTED_DIR, MANIFEST_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CSV_PATH = MANIFEST_DIR / "alimi_manifest.csv"
JSONL_PATH = MANIFEST_DIR / "alimi_manifest.jsonl"

# ---------------------------------------------------------------------------
# Parameters derived from configuration
# ---------------------------------------------------------------------------
DATASET_JSON = (
    CONFIG.dataset_json
    if CONFIG.dataset_json.is_absolute()
    else BASE / CONFIG.dataset_json
)
WHITELIST_PATH = (
    CONFIG.whitelist_file
    if CONFIG.whitelist_file.is_absolute()
    else BASE / CONFIG.whitelist_file
)
YOLO_MODEL = CONFIG.yolo_model
YOLO_CONF_THRESHOLD = CONFIG.yolo_conf_threshold
N_FRAMES_PER_VIDEO = CONFIG.n_frames_per_video
DIVERSITY_THRESHOLD = CONFIG.diversity_threshold
MAX_TRIES_FACTOR = CONFIG.max_tries_factor


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def disk_free_gb(path: pathlib.Path) -> float:
    """Return the free disk space in gigabytes for the filesystem containing *path*."""
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)


def already_downloaded_ids(raw_dir: pathlib.Path = RAW_DIR) -> set:
    return {p.stem for p in raw_dir.glob("*.mp4")}


def fallback_local_videos(raw_dir: pathlib.Path = RAW_DIR) -> List[Dict[str, object]]:
    videos: List[Dict[str, object]] = []
    for path in raw_dir.glob("*.mp4"):
        videos.append(
            {
                "id": path.stem,
                "title": path.stem,
                "webpage_url": "",
                "duration": 10,
                "download_path": str(path),
                "ext": "mp4",
            }
        )
    return videos


def file_is_nonempty(path: pathlib.Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def whitelist_candidates(
    config: PipelineConfig, whitelist_path: pathlib.Path = WHITELIST_PATH
) -> List["Candidate"]:
    if not YTDLP_AVAILABLE or yt_dlp is None:
        logger.error(
            "yt_dlp est requis pour lire la whitelist YouTube. Installez 'yt-dlp'."
        )
        return []
    if not whitelist_path.exists():
        return []

    urls: List[str] = []
    try:
        with whitelist_path.open("r", encoding="utf-8") as fp:
            for raw_line in fp:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                urls.append(line)
    except Exception as exc:
        logger.error("Impossible de lire la whitelist %s: %s", whitelist_path, exc)
        return []

    if not urls:
        return []

    logger.info("Utilisation de la whitelist (%s URLs)", len(urls))

    candidates: List[Candidate] = []
    opts = {
        "quiet": True,
        "skip_download": True,
        "socket_timeout": config.ydl_socket_timeout,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as exc:
                logger.warning("Whitelist URL ignorée (%s): %s", url, exc)
                continue
            video_id = info.get("id") or url
            duration = info.get("duration")
            candidates.append(
                Candidate(
                    id=str(video_id),
                    title=info.get("title", ""),
                    url=info.get("webpage_url", url),
                    duration=duration,
                    ext=info.get("ext"),
                )
            )
    return candidates


@dataclass
class Candidate:
    id: str
    title: str
    url: str
    duration: Optional[float]
    ext: Optional[str]


def fast_search_candidates(config: PipelineConfig) -> List[Candidate]:
    """Search YouTube for videos matching each query with retries."""
    if not YTDLP_AVAILABLE or yt_dlp is None:
        raise RuntimeError("yt_dlp est requis pour interroger YouTube. Installez 'yt-dlp'.")

    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "default_search": "ytsearch",
        "socket_timeout": config.ydl_socket_timeout,
    }

    attempts = config.search_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            candidates: List[Candidate] = []
            with yt_dlp.YoutubeDL(opts) as ydl:
                for query in config.search_queries:
                    search_term = f"ytsearch{config.max_results_per_query}:{query}"
                    logger.info("Recherche YouTube (run=%s): %s", RUN_ID, query)
                    search_result = ydl.extract_info(search_term, download=False)
                    for entry in search_result.get("entries", []) or []:
                        video_id = entry.get("id")
                        if not video_id:
                            continue
                        url = entry.get("url") or entry.get("webpage_url") or video_id
                        if url and not url.startswith("http"):
                            url = f"https://www.youtube.com/watch?v={video_id}"
                        candidates.append(
                            Candidate(
                                id=video_id,
                                title=entry.get("title", ""),
                                url=url,
                                duration=entry.get("duration"),
                                ext=entry.get("ext"),
                            )
                        )
            return candidates
        except Exception as exc:
            if attempt >= attempts:
                logger.error(
                    "Recherche YouTube échouée après %s tentatives: %s", attempt, exc
                )
                raise
            wait_time = config.search_retry_backoff * attempt
            logger.warning(
                "Recherche YouTube échouée (tentative %s/%s): %s | nouvelle tentative dans %.1fs",
                attempt,
                attempts,
                exc,
                wait_time,
            )
            time.sleep(wait_time)
    return []


def _download_single(
    candidate: Candidate, output_template: str, config: PipelineConfig
) -> Optional[Dict[str, object]]:
    if not YTDLP_AVAILABLE or yt_dlp is None:
        raise RuntimeError("yt_dlp est requis pour télécharger des vidéos. Installez 'yt-dlp'.")
    opts = {
        "quiet": True,
        "outtmpl": output_template,
        "noplaylist": True,
        "ignoreerrors": False,
        "socket_timeout": config.ydl_socket_timeout,
    }

    attempts = config.download_retries + 1
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(candidate.url, download=True)
            filepath = pathlib.Path(
                output_template
                % {"id": info["id"], "ext": info.get("ext", candidate.ext or "mp4")}
            )
            if not filepath.exists():
                pattern = filepath.with_suffix(".*")
                found = list(filepath.parent.glob(pattern.name))
                if not found:
                    return None
                filepath = found[0]
            return {
                "id": info.get("id", candidate.id),
                "title": info.get("title", candidate.title),
                "webpage_url": info.get("webpage_url", candidate.url),
                "duration": info.get("duration"),
                "download_path": str(filepath),
                "ext": filepath.suffix.lstrip("."),
            }
        except Exception as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            wait_time = config.download_retry_backoff * attempt
            logger.warning(
                "Téléchargement échoué pour %s (tentative %s/%s): %s | retry dans %.1fs",
                candidate.id,
                attempt,
                attempts,
                exc,
                wait_time,
            )
            time.sleep(wait_time)
    if last_exc is not None:
        raise last_exc
    return None


def probe_and_download(
    candidates: Iterable[Candidate],
    config: PipelineConfig,
    skip_ids: Optional[set] = None,
) -> List[Dict[str, object]]:
    if not YTDLP_AVAILABLE or yt_dlp is None:
        logger.error("yt_dlp est requis pour télécharger des vidéos. Installez 'yt-dlp'.")
        return []
    downloaded: List[Dict[str, object]] = []
    skip_ids = skip_ids or set()
    output_template = str(RAW_DIR / "%(id)s.%(ext)s")

    for candidate in candidates:
        if config.max_videos_per_run is not None and len(downloaded) >= config.max_videos_per_run:
            break
        if candidate.id in skip_ids:
            continue
        duration = candidate.duration
        if duration is not None and (
            duration < config.min_duration or duration > config.max_duration
        ):
            continue
        try:
            logger.info("Téléchargement: %s", candidate.title)
            meta = _download_single(candidate, output_template, config)
            if meta:
                downloaded.append(meta)
        except Exception as exc:
            logger.error("Téléchargement échoué pour %s: %s", candidate.id, exc)
    return downloaded


# ---------------------------------------------------------------------------
# Frame and audio extraction
# ---------------------------------------------------------------------------
def check_disk_space(
    min_gb: float = CONFIG.min_disk_gb_extract, path: pathlib.Path = FRAME_DIR
) -> bool:
    gb = disk_free_gb(path)
    if gb < min_gb:
        logger.error("Espace disque insuffisant (%s Go < %s Go)", gb, min_gb)
        return False
    return True


def extract_audio_wav16(video_path: pathlib.Path, out_wav: pathlib.Path) -> bool:
    if out_wav.exists():
        if file_is_nonempty(out_wav):
            return True
        out_wav.unlink(missing_ok=True)
    if VideoFileClip is None:
        logger.error(
            "moviepy est requis pour l'extraction audio (%s)", MOVIEPY_IMPORT_ERROR
        )
        return False
    clip = None
    try:
        clip = VideoFileClip(str(video_path))
        if clip.audio is None:
            logger.warning("Pas d'audio dans %s", video_path)
            return False
        clip.audio.write_audiofile(
            str(out_wav),
            fps=16000,
            codec="pcm_s16le",
            nbytes=2,
            bitrate=None,
            verbose=False,
            logger=None,
        )
        if file_is_nonempty(out_wav):
            return True
        out_wav.unlink(missing_ok=True)
        return False
    except Exception:
        try:
            if clip:
                clip.close()
        except Exception:
            pass
        try:
            tmp_mp3 = out_wav.with_suffix(".tmp.mp3")
            clip = VideoFileClip(str(video_path))
            if clip.audio is None:
                return False
            clip.audio.write_audiofile(str(tmp_mp3), verbose=False, logger=None)
            clip.close()
            if AudioSegment is None:
                logger.error(
                    "PyDub indisponible pour fallback audio (%s)", PYDUB_IMPORT_ERROR
                )
                tmp_mp3.unlink(missing_ok=True)
                return False
            audio = AudioSegment.from_file(str(tmp_mp3))
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(str(out_wav), format="wav")
            tmp_mp3.unlink(missing_ok=True)
            if file_is_nonempty(out_wav):
                return True
            out_wav.unlink(missing_ok=True)
            return False
        except Exception as exc:
            logger.error("Audio extraction failed for %s: %s", video_path, exc)
            return False
    finally:
        try:
            if clip:
                clip.close()
        except Exception:
            pass


def ahash(img: np.ndarray, hash_size: int = 8) -> np.ndarray:
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("NumPy est requis pour calculer aHash.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = gray.mean()
    bits = (gray > mean_val).astype(np.uint8).flatten()
    return bits


def phash(img: np.ndarray, hash_size: int = 8) -> np.ndarray:
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("NumPy est requis pour calculer pHash.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(gray))
    dct_lowfreq = dct[:hash_size, :hash_size]
    median_val = np.median(dct_lowfreq)
    bits = (dct_lowfreq > median_val).astype(np.uint8).flatten()
    return bits


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("NumPy est requis pour calculer la distance de Hamming.")
    return int(np.sum(a != b))


def _candidate_positions(nfr: int, n: int, margin: float = 0.05) -> List[int]:
    if not NUMPY_AVAILABLE or np is None:
        raise RuntimeError("NumPy est requis pour échantillonner les positions de frames.")
    start = int(margin * nfr)
    end = int((1.0 - margin) * nfr)
    end = max(end, start + n + 1)
    return [int(round(start + (i + 1) / (n + 1) * (end - start))) for i in range(n)]


def extract_n_diverse_frames(
    video_path: pathlib.Path, out_dir: pathlib.Path, n: int = N_FRAMES_PER_VIDEO
) -> List[str]:
    if not NUMPY_AVAILABLE:
        logger.error(
            "Extraction de frames impossible sans NumPy. Installez 'numpy' pour continuer."
        )
        return []
    if not CV2_AVAILABLE:
        logger.error(
            "Extraction de frames impossible sans OpenCV. Installez 'opencv-python' pour continuer."
        )
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        nfr = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if nfr <= 0 or n <= 0:
            return []
        base_positions = _candidate_positions(nfr, n)
        max_tries = max(n * MAX_TRIES_FACTOR, n + 3)
        chosen_paths: List[str] = []
        chosen_ahash: List[np.ndarray] = []
        chosen_phash: List[np.ndarray] = []
        tried = 0
        idx = 0
        while len(chosen_paths) < n and tried < max_tries:
            if idx < len(base_positions):
                pos = base_positions[idx]
                idx += 1
            else:
                pos = int(np.random.randint(0, nfr))
            tried += 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            ah = ahash(frame)
            ph = phash(frame)
            if any(hamming(ah, prev) < DIVERSITY_THRESHOLD for prev in chosen_ahash):
                continue
            if any(hamming(ph, prev) < DIVERSITY_THRESHOLD for prev in chosen_phash):
                continue
            out_jpg = FRAME_DIR / f"{video_path.stem}_{len(chosen_paths):02d}.jpg"
            cv2.imwrite(str(out_jpg), frame)
            if not file_is_nonempty(out_jpg):
                out_jpg.unlink(missing_ok=True)
                continue
            chosen_paths.append(str(out_jpg))
            chosen_ahash.append(ah)
            chosen_phash.append(ph)
        if len(chosen_paths) < n:
            logger.warning("%s: frames extraites < %s (%s)", video_path, n, len(chosen_paths))
        return chosen_paths
    finally:
        capture.release()


def process_one_video(entry: Dict[str, object]) -> Optional[Dict[str, object]]:
    video_path = pathlib.Path(entry.get("download_path", ""))
    if not video_path.exists():
        logger.warning("Fichier vidéo introuvable: %s", video_path)
        return None
    out_wav = AUDIO_DIR / f"{video_path.stem}.wav"
    if out_wav.exists() and not file_is_nonempty(out_wav):
        out_wav.unlink(missing_ok=True)
    frame_list = extract_n_diverse_frames(video_path, FRAME_DIR, N_FRAMES_PER_VIDEO)
    frame_list = [fp for fp in frame_list if file_is_nonempty(pathlib.Path(fp))]
    ok_wav = out_wav.exists() or extract_audio_wav16(video_path, out_wav)
    if ok_wav and frame_list:
        return {**entry, "audio_path": str(out_wav), "frame_paths": frame_list}
    for frame in frame_list:
        try:
            pathlib.Path(frame).unlink()
        except FileNotFoundError:
            pass
    try:
        out_wav.unlink()
    except FileNotFoundError:
        pass
    return None


# ---------------------------------------------------------------------------
# YOLO filtering on extracted frames
# ---------------------------------------------------------------------------
yolo_model: Optional[YOLO] = None


def get_yolo_model() -> YOLO:
    global yolo_model
    if not YOLO_AVAILABLE or YOLO is None:
        raise RuntimeError(
            "Ultralytics YOLO n'est pas disponible. Installez 'ultralytics' pour activer la détection."
        )
    if yolo_model is None:
        try:
            yolo_model = YOLO(YOLO_MODEL)
        except Exception as exc:
            logger.error("Impossible de charger le modèle YOLO %s: %s", YOLO_MODEL, exc)
            raise
    return yolo_model


def frame_has_cat(image_path: str, conf_th: float = YOLO_CONF_THRESHOLD) -> tuple[bool, float]:
    if not YOLO_AVAILABLE or YOLO is None:
        return False, 0.0
    try:
        model = get_yolo_model()
        results = model.predict(source=image_path, verbose=False)
        for result in results:
            raw_names = getattr(result, "names", {})
            if isinstance(raw_names, list):
                names = {idx: name for idx, name in enumerate(raw_names)}
            else:
                names = dict(raw_names)
            if not getattr(result, "boxes", None):
                continue
            for box in result.boxes:
                try:
                    cls_idx = int(box.cls)
                    conf = float(box.conf)
                except (TypeError, ValueError):
                    continue
                if names.get(cls_idx) == "cat" and conf >= conf_th:
                    return True, conf
        return False, 0.0
    except Exception as exc:
        logger.error("YOLO fail on %s: %s", image_path, exc)
        return False, 0.0


def yolo_eval_video(entry: Dict[str, object], conf_th: float = YOLO_CONF_THRESHOLD) -> Optional[Dict[str, object]]:
    if not YOLO_AVAILABLE or YOLO is None:
        enriched = dict(entry)
        enriched.setdefault("yolo_conf", None)
        enriched.setdefault("yolo_hits", 0)
        return enriched
    max_conf = 0.0
    hits = 0
    for frame_path in entry.get("frame_paths", []):
        ok, conf = frame_has_cat(frame_path, conf_th)
        if ok:
            hits += 1
            max_conf = max(max_conf, conf)
    if hits > 0:
        return {**entry, "yolo_conf": round(max_conf, 3), "yolo_hits": hits}

    for frame_path in entry.get("frame_paths", []):
        try:
            pathlib.Path(frame_path).unlink()
        except FileNotFoundError:
            pass
    return None


# ---------------------------------------------------------------------------
# BLIP captioning utilities
# ---------------------------------------------------------------------------
if CAPTIONING_ENABLED:
    blip_model_name = "Salesforce/blip-image-captioning-base"
    try:
        processor = BlipProcessor.from_pretrained(blip_model_name)
        blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_model.to(device)
        logger.info("BLIP chargé sur %s.", device)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Erreur chargement BLIP : %s", exc)
        CAPTIONING_ENABLED = False
        processor = None
        blip_model = None


def caption_image(img_path: str) -> str:
    if not CAPTIONING_ENABLED or processor is None or blip_model is None:
        return ""
    try:
        raw_image = Image.open(img_path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = blip_model.generate(**inputs, max_new_tokens=25)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as exc:
        logger.error("BLIP captioning fail on %s: %s", img_path, exc)
        return ""


def combine_captions(captions: Iterable[str], min_len: int = 3) -> str:
    seen: set[str] = set()
    cleaned: List[str] = []
    for caption in captions:
        c2 = caption.strip().lower()
        if len(c2.split()) >= min_len and c2 not in seen:
            seen.add(c2)
            cleaned.append(caption.strip())
    if len(cleaned) > 5:
        cleaned = cleaned[:5]
    return "; ".join(cleaned) if cleaned else ""


def blip_video_entry(entry: Dict[str, object]) -> Dict[str, object]:
    captions: List[str] = []
    for frame_path in entry.get("frame_paths", []):
        caption = caption_image(frame_path)
        if caption:
            captions.append(caption)
    description = combine_captions(captions)
    return {**entry, "captions": captions, "description": description}


# ---------------------------------------------------------------------------
# Dataset scoring, persistence, and housekeeping
# ---------------------------------------------------------------------------
def quality_score(item: Dict[str, object]) -> float:
    yolo_part = float(item.get("yolo_conf", 0.0) or 0.0)
    score = yolo_part
    if item.get("description"):
        score += 0.1
    return round(max(0.0, min(1.0, score)), 3)


def save_dataset(filtered: Sequence[Dict[str, object]], dataset_json: pathlib.Path = DATASET_JSON) -> List[Dict[str, object]]:
    dataset: List[Dict[str, object]] = []
    for entry in filtered:
        video_path = pathlib.Path(entry.get("download_path", "")) if entry.get("download_path") else None
        item = {
            "id": entry.get("id"),
            "source": "youtube",
            "title": entry.get("title"),
            "duration": entry.get("duration"),
            "video_path": str(video_path) if video_path else None,
            "audio_path": entry.get("audio_path"),
            "frame_paths": entry.get("frame_paths", []),
            "captions": entry.get("captions", []),
            "description": entry.get("description", ""),
            "yolo_conf": round(float(entry.get("yolo_conf", 0.0) or 0.0), 3)
            if entry.get("yolo_conf") is not None
            else None,
            "yolo_hits": int(entry.get("yolo_hits", 0)),
        }
        item["score"] = quality_score(item)
        dataset.append(item)

    dataset.sort(key=lambda data: data["score"], reverse=True)
    with dataset_json.open("w", encoding="utf-8") as fp:
        json.dump(dataset, fp, ensure_ascii=False, indent=2)
    logger.info("Sauvegardé %s items -> %s", len(dataset), dataset_json)
    return dataset


def clean_orphans() -> None:
    valid_video_stems = {path.stem for path in RAW_DIR.glob("*.mp4")}

    def video_stem_from_frame(frame_path: pathlib.Path) -> str:
        stem = frame_path.stem
        return stem.split("_")[0] if "_" in stem else stem

    removed = 0
    for frame_file in list(FRAME_DIR.glob("*.jpg")):
        if video_stem_from_frame(frame_file) not in valid_video_stems:
            try:
                frame_file.unlink()
                removed += 1
            except FileNotFoundError:
                pass

    for audio_file in list(AUDIO_DIR.glob("*.wav")):
        if audio_file.stem not in valid_video_stems:
            try:
                audio_file.unlink()
                removed += 1
            except FileNotFoundError:
                pass

    logger.info("Fichiers orphelins supprimés: %s", removed)


def diagnostics(dataset_json: pathlib.Path = DATASET_JSON) -> None:
    n_raw = len(list(RAW_DIR.glob("*.mp4")))
    n_jpg = len(list(FRAME_DIR.glob("*.jpg")))
    n_wav = len(list(AUDIO_DIR.glob("*.wav")))
    logger.info("🗂️  FICHIERS SUR DISQUE: %s vidéos | %s frames | %s audios", n_raw, n_jpg, n_wav)

    if not dataset_json.exists():
        logger.warning("%s manquant.", dataset_json)
        return

    with dataset_json.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    logger.info("✅ JSON chargé : %s (%s entrées)", dataset_json, len(data))

    scores = [item.get("score") for item in data if item.get("score") is not None]
    if scores:
        logger.info(
            "Score global: n=%s, min=%.3f, mean=%.3f, max=%.3f",
            len(scores),
            min(scores),
            sum(scores) / len(scores),
            max(scores),
        )

    for example in data[:1]:
        summary = {
            key: example[key]
            for key in ("id", "title", "score", "description")
            if key in example
        }
        logger.info("Exemple item (top score): %s", json.dumps(summary, ensure_ascii=False))


def _path_exists(path_str: Optional[str]) -> bool:
    if not path_str:
        return False
    try:
        return pathlib.Path(path_str).exists()
    except OSError:
        return False


def generate_etiq_manifests(
    dataset_json: pathlib.Path = DATASET_JSON,
    csv_path: pathlib.Path = CSV_PATH,
    jsonl_path: pathlib.Path = JSONL_PATH,
) -> List[Dict[str, object]]:
    if not dataset_json.exists():
        logger.error(
            "alim_dataset.json introuvable. Lance d’abord les étapes ALIM (sauvegarde JSON)."
        )
        raise FileNotFoundError("alim_dataset.json introuvable.")

    with dataset_json.open("r", encoding="utf-8") as fp:
        data: List[Dict[str, object]] = json.load(fp)

    usable: List[Dict[str, object]] = []
    for entry in data:
        audio_path = entry.get("audio_path")
        if not _path_exists(audio_path):
            continue
        frames = [frame for frame in entry.get("frame_paths", []) if _path_exists(frame)]
        if not frames:
            continue
        description = (entry.get("description") or "").strip()
        usable.append(
            {
                "id": entry.get("id"),
                "title": entry.get("title"),
                "duration": entry.get("duration"),
                "audio_path": audio_path,
                "description": description,
                "captions_joined": "; ".join(entry.get("captions", []) or []),
                "frame_paths_json": json.dumps(frames, ensure_ascii=False),
                "yolo_conf": entry.get("yolo_conf"),
                "yolo_hits": entry.get("yolo_hits"),
                "score": entry.get("score"),
                "source": entry.get("source", "youtube"),
            }
        )

    logger.info("Items valides pour ETIQ : %s / %s", len(usable), len(data))

    fieldnames = [
        "id",
        "title",
        "duration",
        "audio_path",
        "description",
        "captions_joined",
        "frame_paths_json",
        "yolo_conf",
        "yolo_hits",
        "score",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in usable:
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in usable:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Manifests créés : CSV=%s | JSONL=%s", csv_path, jsonl_path)

    for example in usable[:3]:
        logger.info(
            "—— Aperçu ———\n"
            "id           : %s\n"
            "title        : %s\n"
            "audio_path   : %s\n"
            "frames (n)   : %s\n"
            "desc (début) : %s\n"
            "score        : %s",
            example.get("id"),
            example.get("title"),
            example.get("audio_path"),
            len(json.loads(example.get("frame_paths_json", "[]"))),
            (
                (example.get("description") or "")[:120] + "…"
                if example.get("description")
                and len(example.get("description", "")) > 120
                else example.get("description")
            ),
            example.get("score"),
        )

    return usable


def alim_loop(n_cycles: int = 10, sleep_sec: int = 120) -> None:
    if not NUMPY_AVAILABLE:
        raise RuntimeError(
            "NumPy est requis pour les opérations d'images. Installez 'numpy'."
        )
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV (cv2) est requis pour l'extraction de frames. Installez 'opencv-python'."
        )
    for cycle in range(n_cycles):
        logger.info("==== ALIM CYCLE %s/%s ====", cycle + 1, n_cycles)
        skip_ids = already_downloaded_ids()
        try:
            candidates = fast_search_candidates(CONFIG)
        except Exception as exc:
            logger.error("Recherche YouTube impossible: %s", exc)
            candidates = []
        logger.info("Candidats bruts: %s", len(candidates))
        videos_meta = probe_and_download(candidates, CONFIG, skip_ids=skip_ids)
        if not videos_meta:
            whitelist = whitelist_candidates(CONFIG)
            if whitelist:
                videos_meta = probe_and_download(whitelist, CONFIG, skip_ids=skip_ids)
        logger.info("Candidats téléchargés et filtrés par durée: %s", len(videos_meta))
        if not videos_meta:
            logger.info("Aucun nouveau téléchargement, attente %ss avant prochaine boucle.", sleep_sec)
            time.sleep(sleep_sec)
            continue

        if not check_disk_space():
            logger.error("Arrêt: espace disque insuffisant.")
            break

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            results = list(executor.map(process_one_video, videos_meta))
            prepared_multi = [res for res in results if res]
        logger.info("Paires multi-frames/audio prêtes: %s", len(prepared_multi))

        if YOLO_AVAILABLE and YOLO is not None:
            if not check_disk_space(min_gb=CONFIG.min_disk_gb_yolo):
                logger.error("Arrêt: espace disque insuffisant pour YOLO.")
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                results = list(executor.map(yolo_eval_video, prepared_multi))
                visually_valid = [res for res in results if res]
            logger.info("Validées visuellement: %s", len(visually_valid))
        else:
            logger.warning(
                "YOLO indisponible, saut de la détection : toutes les vidéos préparées sont conservées."
            )
            visually_valid = [dict(entry, yolo_conf=None, yolo_hits=0) for entry in prepared_multi]

        if CAPTIONING_ENABLED and processor and blip_model:
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                visually_valid = list(executor.map(blip_video_entry, visually_valid))
        else:
            for entry in visually_valid:
                entry["captions"] = []
                entry["description"] = ""
        logger.info("Captions générées sur %s vidéos.", len(visually_valid))

        dataset = save_dataset(visually_valid)
        if dataset:
            generate_etiq_manifests()
        clean_orphans()
        diagnostics()
        logger.info("=== Fin du cycle %s ===", cycle + 1)
        time.sleep(sleep_sec)

# ---------------------------------------------------------------------------
# Main pipeline execution
# ---------------------------------------------------------------------------
def run_pipeline() -> List[Dict[str, object]]:
    logger.info("✅ Initialisation et configuration OK")
    if not NUMPY_AVAILABLE:
        raise RuntimeError(
            "NumPy est requis pour les opérations d'images. Installez 'numpy'."
        )
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "OpenCV (cv2) est requis pour l'extraction de frames. Installez 'opencv-python'."
        )
    skip_ids = already_downloaded_ids()
    try:
        candidates = fast_search_candidates(CONFIG)
        logger.info("Candidats bruts: %s", len(candidates))
        videos_meta = probe_and_download(candidates, CONFIG, skip_ids=skip_ids)
        logger.info("Candidats téléchargés et filtrés par durée: %s", len(videos_meta))
    except Exception as exc:
        logger.error("Erreur collecte YouTube: %s", exc)
        videos_meta = []

    if not videos_meta:
        whitelist = whitelist_candidates(CONFIG)
        if whitelist:
            logger.info("Téléchargement via whitelist (%s entrées)", len(whitelist))
            videos_meta = probe_and_download(whitelist, CONFIG, skip_ids=skip_ids)

    if not videos_meta:
        logger.warning(
            "Aucune vidéo YouTube récupérable. Fallback sur vidéos locales présentes."
        )
        videos_meta = fallback_local_videos()

    logger.info("Vidéos détectées (YouTube + local): %s", len(videos_meta))

    if not check_disk_space():
        raise RuntimeError("Espace disque insuffisant pour extraction.")

    logger.info(
        "Extraction audio+multi-frames pour %s vidéos (parallélisé)...", len(videos_meta)
    )
    prepared_multi: List[Dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        results = list(executor.map(process_one_video, videos_meta))
        prepared_multi = [res for res in results if res]
    logger.info("Paires multi-frames/audio prêtes: %s", len(prepared_multi))

    if not prepared_multi:
        logger.warning("Aucune donnée préparée, manifeste non créé.")
        return []

    if YOLO_AVAILABLE and YOLO is not None:
        if not check_disk_space(min_gb=CONFIG.min_disk_gb_yolo):
            raise RuntimeError("Espace disque insuffisant pour YOLO.")

        logger.info(
            "Détection YOLO sur %s vidéos (parallélisé)...", len(prepared_multi)
        )
        visually_valid: List[Dict[str, object]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            results = list(executor.map(yolo_eval_video, prepared_multi))
            visually_valid = [res for res in results if res]
        logger.info("Validées visuellement (≥1 frame chat): %s", len(visually_valid))
    else:
        logger.warning(
            "YOLO indisponible, saut de la détection : toutes les vidéos préparées sont conservées."
        )
        visually_valid = [dict(entry, yolo_conf=None, yolo_hits=0) for entry in prepared_multi]

    if not visually_valid:
        logger.warning("Aucune vidéo validée visuellement, manifeste non créé.")
        return []

    if CAPTIONING_ENABLED and processor and blip_model:
        logger.info(
            "BLIP: génération captions pour %s vidéos (parallélisé)...",
            len(visually_valid),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            visually_valid = list(executor.map(blip_video_entry, visually_valid))
        n_with_desc = sum(1 for entry in visually_valid if entry.get("description"))
        logger.info(
            "Descriptions générées pour %s/%s vidéos.",
            n_with_desc,
            len(visually_valid),
        )
    else:
        for entry in visually_valid:
            entry["captions"] = []
            entry["description"] = ""
        logger.info("BLIP désactivé, captions vides.")

    logger.info(
        "Exemple captions: %s",
        visually_valid[0]["captions"] if visually_valid else "Aucun",
    )

    dataset = save_dataset(visually_valid)
    if dataset:
        generate_etiq_manifests()
    clean_orphans()
    diagnostics()
    return dataset


if __name__ == "__main__":
    run_pipeline()
