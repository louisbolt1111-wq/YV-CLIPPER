import streamlit as st
import os
import json
import time
import tempfile
import http.cookiejar
import requests
from urllib.parse import urlparse, parse_qs
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoFileClip

# --- Page Configuration ---
st.set_page_config(page_title="AI Viral Shorts Clipper", page_icon="✂️", layout="centered")
st.title("✂️ AI Viral YouTube Clipper")
st.write("Extract the most engaging, viral moments from long videos and convert them into vertical Shorts.")

# --- Structured Output Schema for Gemini ---
class ViralClip(BaseModel):
    title: str = Field(description="Catchy viral title for the Short")
    start_seconds: float = Field(description="Exact start time in seconds")
    end_seconds: float = Field(description="Exact end time in seconds (clip should be 20 to 55 seconds long)")
    reason: str = Field(description="Why this segment has high viral potential")

class ClipList(BaseModel):
    clips: list[ViralClip]

# --- Helper Functions ---
def extract_video_id(url: str):
    parsed = urlparse(url)
    if parsed.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed.path.lstrip('/').split('/')[0]
    if parsed.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        if parsed.path.startswith(('/embed/', '/v/', '/shorts/')):
            parts = parsed.path.split('/')
            return parts[2] if len(parts) > 2 else None
    return None

def get_youtube_transcript_with_fallback(video_id: str, url: str, cookie_path: str = None, proxy_url: str = None) -> str:
    # Method 1: Try youtube-transcript-api first
    try:
        session = requests.Session()
        if proxy_url:
            session.proxies = {"http": proxy_url, "https": proxy_url}
        if cookie_path and os.path.exists(cookie_path):
            cj = http.cookiejar.MozillaCookieJar(cookie_path)
            cj.load(ignore_discard=True, ignore_expires=True)
            session.cookies = cj

        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, proxies=proxies, cookies=cookie_path)
        else:
            ytt_api = YouTubeTranscriptApi(http_client=session)
            fetched = ytt_api.fetch(video_id)
            transcript_list = fetched.to_raw_data() if hasattr(fetched, 'to_raw_data') else list(fetched)

        formatted = []
        for entry in transcript_list:
            start = round(entry['start'] if isinstance(entry, dict) else entry.start, 1)
            raw_text = entry['text'] if isinstance(entry, dict) else entry.text
            text = raw_text.replace('\n', ' ')
            formatted.append(f"[{start}s] {text}")
        if formatted:
            return "\n".join(formatted)
    except Exception:
        pass

    # Method 2: Fallback to yt-dlp subtitle extraction (much better with cookies.txt)
    with tempfile.TemporaryDirectory() as sub_dir:
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'subtitlesformat': 'json3',
            'outtmpl': os.path.join(sub_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        if cookie_path and os.path.exists(cookie_path):
            ydl_opts['cookiefile'] = cookie_path
        if proxy_url:
            ydl_opts['proxy'] = proxy_url

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for fname in os.listdir(sub_dir):
            if fname.endswith('.json3'):
                with open(os.path.join(sub_dir, fname), 'r', encoding='utf-8') as f:
                    sub_data = json.load(f)
                formatted = []
                for event in sub_data.get('events', []):
                    if 'segs' in event:
                        start_sec = round(event.get('tStartMs', 0) / 1000.0, 1)
                        text = "".join(seg.get('utf8', '') for seg in event['segs']).replace('\n', ' ').strip()
                        if text:
                            formatted.append(f"[{start_sec}s] {text}")
                if formatted:
                    return "\n".join(formatted)

    return ""

def download_youtube_video(url: str, output_path: str, cookie_path: str = None, proxy_url: str = None):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    if cookie_path and os.path.exists(cookie_path):
        ydl_opts['cookiefile'] = cookie_path
    if proxy_url:
        ydl_opts['proxy'] = proxy_url

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def analyze_with_gemini_rotation(api_keys: list[str], content_source, is_video_file: bool = False) -> list[ViralClip]:
    prompt = (
        "Analyze this content and find the top 3 most engaging, high-retention, viral segments suitable for YouTube Shorts. "
        "Each clip MUST be between 20 and 55 seconds long. Look for strong hooks, surprising facts, emotional stories, or actionable advice. "
        "Return exact start_seconds and end_seconds."
    )
    
    models_to_try = [
        "gemini-3.8-flash-lite",
        "gemini-3.8-flash",
        "gemini-3.0-flash",
        "gemini-2.5-flash-lite",
    ]
    
    last_error = None
    for index, key in enumerate(api_keys):
        client = genai.Client(api_key=key)
        uploaded_file = None
        
        if is_video_file:
            uploaded_file = client.files.upload(file=content_source)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(3)
                uploaded_file = client.files.get(name=uploaded_file.name)
            if uploaded_file.state.name != "ACTIVE":
                raise RuntimeError(f"Gemini video processing failed with state: {uploaded_file.state.name}")

        for model_name in models_to_try:
            try:
                contents = [uploaded_file, prompt] if is_video_file else f"{prompt}\n\nTRANSCRIPT:\n{content_source}"
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ClipList,
                    ),
                )
                data = json.loads(response.text)
                return [ViralClip(**item) for item in data.get("clips", [])]
                
            except Exception as e:
                last_error = e
                error_msg = str(e).lower()
                if "429" in error_msg or "quota" in error_msg or "resource_exhausted" in error_msg:
                    st.warning(f"Key #{index + 1} hit quota limit. Switching to next key...")
                    break
                elif "503" in error_msg or "unavailable" in error_msg or "404" in error_msg or "not_found" in error_msg:
                    st.info(f"Model `{model_name}` busy/unavailable. Trying backup model...")
                    continue
                else:
                    raise e
                
    raise RuntimeError(f"All provided Gemini API keys or models failed. Last error: {last_error}")

def cut_and_format_clip(video_path: str, start: float, end: float, output_path: str, make_vertical: bool):
    with VideoFileClip(video_path) as video:
        max_duration = video.duration
        start = max(0, min(start, max_duration - 5))
        end = min(end, max_duration)
        
        subclip = video.subclip(start, end)
        
        if make_vertical:
            w, h = subclip.size
            target_ratio = 9 / 16
            new_w = int(h * target_ratio)
            if new_w < w:
                x_center = w / 2
                subclip = subclip.crop(x1=x_center - new_w / 2, x2=x_center + new_w / 2)
        
        subclip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            logger=None
        )

# --- Sidebar Settings ---
with st.sidebar:
    st.header("🔑 Flexible Gemini API Keys")
    st.caption("Paste one key, or multiple keys (one per line). Change or replace anytime without redeploying.")
    
    default_key = st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else ""
    raw_keys_input = st.text_area(
        "Enter Gemini API Key(s)",
        value=default_key,
        height=100,
        placeholder="AQ.Ab8RN6... or AIzaSy...",
        help="Get free keys from Google AI Studio (aistudio.google.com)."
    )
    
    api_keys = [k.strip() for k in raw_keys_input.splitlines() if k.strip()]
    if api_keys:
        st.success(f"✅ {len(api_keys)} API key(s) loaded and ready.")
    else:
        st.warning("⚠️ No API key entered yet.")

    st.markdown("---")
    st.header("🍪 YouTube Unblock Settings")
    st.caption("Upload cookies.txt (from a burner account) and/or enter a proxy URL to bypass Streamlit Cloud IP blocks.")
    cookie_file = st.file_uploader("Upload cookies.txt (Optional)", type=["txt"])
    if cookie_file:
        st.success("✅ Cookies loaded for YouTube requests.")

    proxy_url = st.text_input(
        "Proxy URL (Optional)",
        placeholder="http://user:pass@host:port",
        help="Optional residential proxy if YouTube blocks cloud IPs even with cookies."
    )

    st.markdown("---")
    st.header("🎬 Video Settings")
    make_vertical = st.toggle("Crop to 9:16 Vertical (Shorts)", value=True)

# --- Main Interface ---
tab1, tab2 = st.tabs(["🔗 Paste YouTube Link", "📁 Upload Video File (Backup)"])

with tab1:
    yt_url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
    run_yt = st.button("✨ Generate Viral Clips from Link", type="primary", use_container_width=True)

with tab2:
    uploaded_video = st.file_uploader("Upload an MP4 video", type=["mp4", "mov", "mkv"])
    run_upload = st.button("✨ Generate Viral Clips from File", type="primary", use_container_width=True)

# --- Processing Logic ---
if run_yt or run_upload:
    if not api_keys:
        st.error("Please paste at least one Gemini API Key in the sidebar first.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_video_path = os.path.join(tmpdir, "source.mp4")
        cookie_path = None
        if cookie_file:
            cookie_path = os.path.join(tmpdir, "cookies.txt")
            with open(cookie_path, "wb") as cf:
                cf.write(cookie_file.getvalue())
        
        try:
            if run_yt:
                if not yt_url:
                    st.warning("Please paste a valid YouTube link.")
                    st.stop()
                
                video_id = extract_video_id(yt_url)
                if not video_id:
                    st.error("Could not extract a valid YouTube Video ID from that URL.")
                    st.stop()

                with st.status("Step 1/3: Downloading video & transcript...", expanded=True) as status:
                    st.write("Downloading video stream...")
                    download_youtube_video(yt_url, raw_video_path, cookie_path=cookie_path, proxy_url=proxy_url.strip() or None)
                    st.write("Extracting transcript...")
                    transcript_text = get_youtube_transcript_with_fallback(
                        video_id, yt_url, cookie_path=cookie_path, proxy_url=proxy_url.strip() or None
                    )
                    status.update(label="Download complete!", state="complete")

                with st.status("Step 2/3: AI is hunting for viral moments...", expanded=True) as status:
                    if transcript_text:
                        clips = analyze_with_gemini_rotation(api_keys, transcript_text, is_video_file=False)
                    else:
                        st.info("No text transcript available — sending downloaded video directly to Gemini AI...")
                        clips = analyze_with_gemini_rotation(api_keys, raw_video_path, is_video_file=True)
                    status.update(label=f"Found {len(clips)} viral segments!", state="complete")

            elif run_upload:
                if not uploaded_video:
                    st.warning("Please upload a video file first.")
                    st.stop()
                
                with open(raw_video_path, "wb") as f:
                    f.write(uploaded_video.read())

                with st.status("Step 1/2: Uploading video to Gemini AI for analysis...", expanded=True) as status:
                    clips = analyze_with_gemini_rotation(api_keys, raw_video_path, is_video_file=True)
                    status.update(label=f"Found {len(clips)} viral segments!", state="complete")

            if not clips:
                st.warning("No viral clips were identified. Try a different video.")
                st.stop()

            st.subheader("🔥 Your Viral Shorts")
            for idx, clip in enumerate(clips, start=1):
                with st.container(border=True):
                    st.markdown(f"### #{idx}: {clip.title}")
                    st.caption(f"⏱️ Timestamp: `{clip.start_seconds}s` → `{clip.end_seconds}s` | 💡 **Why it works:** {clip.reason}")
                    
                    out_file = os.path.join(tmpdir, f"short_{idx}.mp4")
                    with st.spinner(f"Rendering clip #{idx}..."):
                        cut_and_format_clip(raw_video_path, clip.start_seconds, clip.end_seconds, out_file, make_vertical)
                    
                    with open(out_file, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                        st.download_button(
                            label=f"⬇️ Download Short #{idx}",
                            data=video_bytes,
                            file_name=f"viral_short_{idx}.mp4",
                            mime="video/mp4",
                            key=f"dl_{idx}"
                        )

        except Exception as e:
            st.error(f"An error occurred: {e}")
            if "Sign in to confirm" in str(e) or "403" in str(e) or "blocking requests" in str(e):
                st.info("💡 Tip: Make sure you exported cookies while logged into YouTube, or add a Proxy URL in the sidebar, or use the **Upload Video File (Backup)** tab!")
