import os
import time
import queue
import threading

import pyaudio
import pygame
import whisper
import numpy as np
from numpy.linalg import norm
import torch
# from pydub import AudioSegment
# from pydub.playback import play
from sentence_transformers import SentenceTransformer

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.clock import Clock


def cosine_similarity(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))


def trigger_detected(model, sentence, refs_embs):
    t0 = time.time()
    sentence_embs = model.encode(sentence)
    for embs in refs_embs:
        if cosine_similarity(sentence_embs, embs) > 0.8:
            # print(time.time() - t0)
            return True
    # print(time.time() - t0)
    return False


class SpeechNode(Node):
    def __init__(self):
        super().__init__('speech_node')

        # Define audio parameters
        self.CHUNK = 4096//4  # Samples to read per frame
        self.FORMAT = pyaudio.paInt16  # Audio format (16-bit signed PCM)
        self.CHANNELS = 1  # Mono audio
        self.RATE = 16000  # Sampling rate (Hz)
        self.RECORD_SECONDS = 3  # Duration of recording in seconds
        self.STOP_TALKING_THRESHOLD = 1.5  # Duration of silence to stop talking

        #  FIXME:  CHANGE THESE DEPENDING ON THE MIC
        self.THRESH_FREQ_MIN = 90
        self.THRESH_FREQ_MAX = 400
        self.SUM_SIG = 200

        logging.info("Model: Loading")
        self.whisper_model = whisper.load_model("base.en")
        self.mini_lm_model = SentenceTransformer('paraphrase-MiniLM-L6-v2',
                                                 device='cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Model: Loaded")

        # Keywords for wake word detection
        self.pepper_detection_keywords = ["Robot", "Hey Robot", "Hey Robot !",
                                          'Hey Pepper !', "Pepper !", "Pepper", "Hey Pepper", "Hey Paper !", "Pippa",
                                          "Hey Pippa", "Hey Pippa !", "Pippa !", "Bippo", "Bipper", "paper",
                                          "Hey Tiago", "Hey Tiago !", "Tiago !", "Tigo", "Diego", "Hey Diego",
                                          ]
        self.embed_pepper_keywords = self.mini_lm_model.encode(self.pepper_detection_keywords)
        self.siri_filename = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets", "siri-3.wav"))

        self.wake_word_detection = True
        self.I_am_listening = False
        self.last_conversation_end = None

        self.audio_queue = queue.Queue()
        # self.race_condition = threading.Condition()
        self.speak_mutex = threading.Lock()

        # ROS
        self.ros_clock = Clock()  # TODO
        self.speech_pub = self.create_publisher(String, '/speech/text', 1)
        self.trigger_detected_pub = self.create_publisher(String, '/speech/trigger_detected', 1)
        self.tts_sub = self.create_subscription(String, '/speech/say', self.say_text, 1)

    def say_text(self, text_msg):  # [String, str], method='gtts', wait_until_finish=True):
        if isinstance(text_msg, String):
            text = text_msg.data
        else:
            text = text_msg

        print(f"speech.say_text(): {text}")
        method = 'gtts'
        wait_until_finish = True
        self.speak_mutex.acquire()
        gtts_failed = False
        if method.lower() == 'gtts':
            try:
                from gtts import gTTS
                mp3_file = "/tmp/output.mp3"
                tts = gTTS(text)
                tts.save(mp3_file)

                # use pygame to play mp3 file
                pygame.mixer.init()
                pygame.mixer.music.load(mp3_file)
                pygame.mixer.music.play()

                if wait_until_finish:
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)

            except Exception as e:
                gtts_failed = True
                print(f"Error: {e}")

        if gtts_failed or method.lower() != 'gtts':
            try:
                import pyttsx3
                engine = pyttsx3.init()  # Initialize the engine
                engine.setProperty("rate", 120)
                engine.setProperty("volume", 0.9)
                engine.setProperty("voice", "english-us")
                engine.say(text)
                if wait_until_finish:
                    engine.runAndWait()

            except Exception as e:
                print(f"Error: {e}")

        self.last_conversation_end = time.time()  # rclpy.clock.Clock().now()
        self.speak_mutex.release()

    def audio_read(self, name):
        logging.info("Audio reading: starting")
        au = pyaudio.PyAudio()
        stream = au.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        logging.info("Audio reading: Started")

        is_talking = False
        start_silence = None
        frames = []
        while True:
            data = stream.read(self.CHUNK)
            signal = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            fft = np.fft.fft(signal)
            max_freq = (self.RATE / self.CHUNK) * np.argmax(np.abs(fft))
            is_currently_talking = np.sum(np.abs(fft)) > self.SUM_SIG and (
                        self.THRESH_FREQ_MIN < max_freq < self.THRESH_FREQ_MAX)

            if is_currently_talking:
                is_talking = True
                start_silence = None  # Reset silence timer when talking starts
                frames.append(data)
            elif is_talking:
                frames.append(data)  # Append data to frames
                # This block only enters if previously talking, now silent
                if start_silence is None:
                    start_silence = time.time()  # Start counting silence time

                if time.time() - start_silence > self.STOP_TALKING_THRESHOLD:
                    self.audio_queue.put(frames)  # Send frames to queue

                    frames = []  # Reset frames
                    is_talking = False  # Reset talking flag
                    start_silence = None  # Reset silence timer

            # Check if it's been more than 5 seconds since last conversation ended
            if self.last_conversation_end and not is_currently_talking and (time.time() - self.last_conversation_end > 5):
                print("No talking for more than 5 seconds since last conversation ended.")
                self.last_conversation_end = None  # Reset last conversation end timer to avoid repeated messages

    def exec(self):
        thread_listening = threading.Thread(target=self.audio_read, args=(1,), daemon=True)
        thread_listening.start()

        while True:
            if self.audio_queue.empty():
                time.sleep(0.1)
                continue
            audio_data = np.frombuffer(b''.join(self.audio_queue.get()), dtype=np.int16).astype(np.float32) / 32768.0
            cp = audio_data.copy()
            try:
                result = self.whisper_model.transcribe(audio=cp, fp16=torch.cuda.is_available())
            except Exception as e:
                print(f"Error: {e}")
                continue

            print(f"detection: {result['text']}")
            if self.I_am_listening is True:
                msg = String(data=result["text"])
                self.speech_pub.publish(msg)
            else:
                logging.info("I am not listening")

            if (self.wake_word_detection is True and self.I_am_listening is False and
                    trigger_detected(self.mini_lm_model, result["text"], self.embed_pepper_keywords)):
                logging.info("Trigger detected")
                msg_trig = String(data="trigger_detected")
                self.trigger_detected_pub.publish(msg_trig)
                self.I_am_listening = True
                # siri = AudioSegment.from_wav(self.siri_filename)
                # play(siri)
                pygame.mixer.init()
                pygame.mixer.music.load(self.siri_filename)
                pygame.mixer.music.play()


if __name__ == "__main__":
    import logging
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
    rclpy.init()
    speech_node = SpeechNode()
    speech_node.exec()
    rclpy.spin(speech_node)
    speech_node.destroy_node()
    rclpy.shutdown()
