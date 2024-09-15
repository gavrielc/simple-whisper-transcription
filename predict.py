# Prediction interface for Cog ⚙️
from typing import Any, List
import datetime
import subprocess
import os
import time
import torch
import re

from cog import BasePredictor, BaseModel, Input, File, Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


class Output(BaseModel):
    transcription: str


def format_speech(speech_list):
    def format_time(seconds):
        # Convert the float to total seconds and then compute minutes and seconds
        total_seconds = int(float(seconds))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        # Format as m:ss
        return f"{minutes}:{seconds:02}"

    formatted_string = ""
    for item in speech_list:
        # Format the start time
        start_time = format_time(item['start'])
        formatted_string += f"{item['speaker']}\n[{start_time}]\n{item['text']}\n\n"
    return formatted_string


class Predictor(BasePredictor):

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_name = "large-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type="float16" if device == "cuda" else "float32",
        )
        self.diarization_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token="YOUR_HUGGINGFACE_TOKEN",
        ).to(torch.device(device))

    def predict(
        self,
        file: Path = Input(description="Audio file", default=None),
        prompt: str = Input(
            description="Vocabulary: provide names, acronyms, and loanwords in a list. Use punctuation for best accuracy.",
            default=None,
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        try:
            # Generate a temporary filename
            temp_wav_filename = f"temp-{time.time_ns()}.wav"
            offset_seconds = 0
            group_segments = True
            language = None
            num_speakers = None

            if file is not None:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        str(file),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        temp_wav_filename,
                    ],
                    check=True,
                )

            segments, detected_num_speakers, detected_language = self.speech_to_text(
                temp_wav_filename,
                num_speakers,
                prompt,
                offset_seconds,
                group_segments,
                language,
                word_timestamps=True,
                transcript_output_format="segments_only",
            )

            print(f"Done with inference")
            print(
                f"Number of Speakers: {detected_num_speakers}. Detected language: {detected_language}."
            )
            transcription = format_speech(segments)
            # Return the results as a JSON object
            return Output(
                transcription=transcription,
            )

        except Exception as e:
            raise RuntimeError("Error running inference with local model") from e

        finally:
            # Clean up
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        offset_seconds=0,
        group_segments=True,
        language=None,
        word_timestamps=True,
        transcript_output_format="both",
    ):
        time_start = time.time()

        # Transcribe audio
        print("Starting transcription")
        options = dict(
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=1000),
            initial_prompt=prompt,
            word_timestamps=word_timestamps,
            language=language,
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start + offset_seconds),
                "end": float(s.end + offset_seconds),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start + offset_seconds),
                        "end": float(w.end + offset_seconds),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]

        time_transcribing_end = time.time()
        print(
            f"Finished transcription, took {time_transcribing_end - time_start:.5} seconds"
        )

        print("Starting diarization")
        diarization = self.diarization_model(audio_file_wav, num_speakers=num_speakers)

        time_diarization_end = time.time()
        print(
            f"Finished diarization, took {time_diarization_end - time_transcribing_end:.5} seconds"
        )

        print("Starting speaker assignment")

        # Assign speakers to words
        margin = 0.1  # 0.1 seconds margin
        words_with_speakers = []

        diarization_list = list(diarization.itertracks(yield_label=True))
        unique_speakers = {
            speaker for _, _, speaker in diarization.itertracks(yield_label=True)
        }
        detected_num_speakers = len(unique_speakers)

        # Sort diarization segments by start time
        diarization_list.sort(key=lambda x: x[0].start)
        n_diarization_segments = len(diarization_list)
        diarization_idx = 0

        # Iterate over each word in all segments
        for segment in segments:
            for word in segment["words"]:
                word_start = word["start"] - margin
                word_end = word["end"] + margin

                # Find the speaker for this word
                while diarization_idx < n_diarization_segments:
                    turn, _, speaker = diarization_list[diarization_idx]

                    if turn.end < word_start:
                        # Move to the next diarization segment
                        diarization_idx += 1
                    elif turn.start > word_end:
                        # Word occurs before the current diarization segment
                        break
                    else:
                        # Word overlaps with speaker turn
                        word_with_speaker = {
                            "start": word["start"],
                            "end": word["end"],
                            "word": word["word"].strip(),
                            "speaker": speaker,
                        }
                        words_with_speakers.append(word_with_speaker)
                        break
                else:
                    # No more diarization segments; assign 'Unknown' speaker
                    word_with_speaker = {
                        "start": word["start"],
                        "end": word["end"],
                        "word": word["word"].strip(),
                        "speaker": "Unknown",
                    }
                    words_with_speakers.append(word_with_speaker)

        time_assignment_end = time.time()
        print(
            f"Finished speaker assignment, took {time_assignment_end - time_diarization_end:.5} seconds"
        )

        print("Starting segment building")

        # Build new segments based on speaker changes
        new_segments = []

        if words_with_speakers:
            current_speaker = words_with_speakers[0]["speaker"]
            current_segment_words = [words_with_speakers[0]]
            current_segment_start = words_with_speakers[0]["start"]
            current_segment_end = words_with_speakers[0]["end"]

            for word_info in words_with_speakers[1:]:
                word_speaker = word_info["speaker"]
                if word_speaker == current_speaker:
                    # Continue current segment
                    current_segment_words.append(word_info)
                    current_segment_end = word_info["end"]
                else:
                    # Save current segment
                    segment_text = ' '.join([w['word'] for w in current_segment_words])
                    new_segment = {
                        "start": current_segment_start,
                        "end": current_segment_end,
                        "speaker": current_speaker,
                        "text": segment_text.strip(),
                        "words": current_segment_words,
                    }
                    new_segments.append(new_segment)

                    # Start a new segment
                    current_speaker = word_speaker
                    current_segment_words = [word_info]
                    current_segment_start = word_info["start"]
                    current_segment_end = word_info["end"]

            # Add the last segment
            segment_text = ' '.join([w['word'] for w in current_segment_words])
            new_segment = {
                "start": current_segment_start,
                "end": current_segment_end,
                "speaker": current_speaker,
                "text": segment_text.strip(),
                "words": current_segment_words,
            }
            new_segments.append(new_segment)

        # Optionally merge segments with small time gaps
        final_segments = []

        if new_segments:
            current_group = new_segments[0]

            for segment in new_segments[1:]:
                time_gap = segment["start"] - current_group["end"]
                if (
                    segment["speaker"] == current_group["speaker"]
                    and time_gap <= 2
                    and group_segments
                ):
                    # Merge segments
                    current_group["end"] = segment["end"]
                    current_group["text"] += " " + segment["text"]
                    current_group["words"].extend(segment["words"])
                else:
                    # Add current group to final_segments
                    final_segments.append(current_group)
                    # Start a new group
                    current_group = segment

            # Add the last group
            final_segments.append(current_group)

        time_segment_end = time.time()
        print(
            f"Finished segment building, took {time_segment_end - time_assignment_end:.5} seconds"
        )
        time_end = time.time()
        time_diff = time_end - time_start

        system_info = f"""Processing time: {time_diff:.5} seconds"""
        print(system_info)
        return final_segments, detected_num_speakers, transcript_info.language
