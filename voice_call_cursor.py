import speech_recognition as sr
import os
from dotenv import load_dotenv
from langgraph.checkpoint.mongodb import MongoDBSaver
from graph import create_chat_graph
import soundfile as sf
from dia.model import Dia

# Load environment variables
load_dotenv()
MONGODB_URI = "mongodb://localhost:27017/"
CONFIG = {"configurable": {"thread_id": "1"}}

# Load Dia model once at startup
dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B")

def main():
    # Initialize MongoDB checkpointer
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph = create_chat_graph(checkpointer=checkpointer)
        r = sr.Recognizer()
        r.dynamic_energy_threshold = True
        r.pause_threshold = 2

        with sr.Microphone() as source:
            print("Adjusting for ambient noise. Please wait...")
            r.adjust_for_ambient_noise(source)
            print("Ready to receive voice input.")

            while True:
                print("\nPlease say something:")
                audio = r.listen(source)

                try:
                    print("Recognizing...")
                    sst = r.recognize_google(audio)  # You can set language='en-IN' if needed
                    print("You said:", sst)
                except sr.UnknownValueError:
                    print("❌ Could not understand audio. Please try again.")
                    continue
                except sr.RequestError as e:
                    print(f"❌ Google Speech Recognition request failed: {e}")
                    continue

                # Get response from chat graph
                response_text = ""
                try:
                    for event in graph.stream(
                        {"messages": [{"role": "user", "content": sst}]},
                        CONFIG,
                        stream_mode="values"
                    ):
                        if "messages" in event:
                            for msg in event["messages"]:
                                print(msg)
                                response_text += str(msg)
                except Exception as e:
                    print(f"❌ Error in generating response from graph: {e}")
                    continue

                # Generate audio from Dia model
                if response_text.strip():
                    try:
                        print("🔊 Generating speech with Dia...")
                        dia_audio = dia_model.generate(response_text)
                        sf.write("response.wav", dia_audio, 44100)
                        print("✅ Audio response saved as response.wav")

                        # Optional: play the audio
                        # import sounddevice as sd
                        # sd.play(dia_audio, 44100)
                        # sd.wait()
                    except Exception as e:
                        print(f"❌ Error generating speech audio: {e}")


main()
