import azure.cognitiveservices.speech as speechsdk


class TextToSpeechService:
    def __init__(self, speech_key: str, speech_region: str):
        # 設定地區和 key
        self.speech_config = speechsdk.SpeechConfig(
            subscription=speech_key, region=speech_region
        )

    def text_to_wav(self, text: str):
        # 設定語言
        self.speech_config.speech_synthesis_voice_name = "zh-TW-HsiaoChenNeural"

        # 使用預設聲音輸出的範例
        # audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        # speech_synthesizer = speechsdk.SpeechSynthesizer(self.speech_config=speech_config, audio_config=audio_config)

        # 使用檔案輸出
        file_name = "media/response.wav"
        file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=file_config
        )
        result = speech_synthesizer.speak_text_async(text).get()

        # 檢查結果偵錯
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesized for text [{}]".format(text))
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
