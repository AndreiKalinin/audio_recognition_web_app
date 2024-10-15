import requests
import uuid
import pandas as pd
import json
import io
from pydub import AudioSegment
from mutagen.mp3 import MP3


def time_format(sec):
    m = str(int(sec // 60))
    s = str(int(sec % 60))
    if len(s) == 1:
        return m + ':0' + s
    return m + ':' + s


class AudioTranscriber:
    """This class receives audio file URL and transforms audio into text using SaluteSpeech"""
    def __init__(self, audio_provider_token, cert_path):
        self.audio_provider_token = audio_provider_token
        self.cert_path = cert_path
        self.token = None

    def get_token(self):
        """Get token. Token duration is 30 min"""
        auth_data = self.audio_provider_token
        oath_url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
        auth_headers = {
            'Authorization': 'Basic ' + auth_data,
            'RqUID': str(uuid.uuid4()),
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        scope_params = {
            'scope': 'SALUTE_SPEECH_PERS'
        }
        token_respond = requests.post(oath_url,
                                      headers=auth_headers,
                                      data=scope_params,
                                      verify=self.cert_path)
        self.token = token_respond.json()['access_token']


    def upload_file(self, link):
        """Send file into SaluteSpeech cloud and get file id"""
        api_url = 'https://smartspeech.sber.ru/rest/v1/data:upload'

        request_headers = {
            'Authorization': 'Bearer ' + self.token
        }

        yd_base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

        final_url = yd_base_url + 'public_key=' + link
        response = requests.get(final_url)
        download_url = response.json()['href']

        r = requests.get(download_url)
        input_file_buffer = io.BytesIO(r.content)
        output_file_buffer = io.BytesIO()

        AudioSegment.from_file(input_file_buffer).export(output_file_buffer, format='mp3')
        output_file = MP3(output_file_buffer)
        num_channels = output_file.info.channels
        sample_rate = output_file.info.sample_rate
        file_length = output_file.info.length

        data_respond = requests.post(api_url,
                                     headers=request_headers,
                                     data=output_file_buffer,
                                     verify=self.cert_path)
        input_file_buffer.close()
        output_file_buffer.close()
        return data_respond.json()['result']['request_file_id'], num_channels, sample_rate, file_length


    def send_transcription_task(self, request_id, sample_rate, num_channels):
        """Send audio recognition task"""
        api_url = 'https://smartspeech.sber.ru/rest/v1/speech:async_recognize'

        request_headers = {
            'Authorization': 'Bearer ' + self.token,
            'Content-Type': 'application/json'
        }

        post_data = {
            'options': {
                'audio_encoding': 'MP3',
                'sample_rate': sample_rate,
                'channels_count': num_channels,
                'language': 'ru-RU',
                'model': 'callcenter',
            },
            'request_file_id': request_id
        }

        data_respond = requests.post(api_url,
                                     headers=request_headers,
                                     data=json.dumps(post_data),
                                     verify=self.cert_path)
        return data_respond.json()['result']['id']


    def check_status(self, task_id):
        """Check task state and return result id"""
        api_url = 'https://smartspeech.sber.ru/rest/v1/task:get'

        request_headers = {
            'Authorization': 'Bearer ' + self.token,
        }

        params = {
            'id': task_id
        }

        respond = requests.get(api_url,
                               headers=request_headers,
                               params=params,
                               verify=self.cert_path)
        res = respond.json()['result']
        return res


    def download_result(self, result_id):
        """Download audio recognition result and transform into dataframe"""
        api_url = 'https://smartspeech.sber.ru/rest/v1/data:download'

        request_headers = {
            'Authorization': 'Bearer ' + self.token,
        }

        params = {
            'response_file_id': result_id
        }

        respond = requests.get(api_url,
                               headers=request_headers,
                               params=params,
                               verify=self.cert_path)
        df = pd.DataFrame(respond.json())
        for col in ['results', 'emotions_result', 'speaker_info', 0, 'backend_info']:
            df[col] = df[col].map(lambda x: eval(f'{x}') if pd.notnull(x) else x)
            df = pd.concat([df, df.pop(col).apply(pd.Series)], axis=1)
        df = df[['normalized_text', 'start', 'end', 'positive', 'neutral', 'negative']]
        df['start'] = df['start'].str.replace('s', '').astype(float).apply(time_format)
        df['end'] = df['end'].str.replace('s', '').astype(float).apply(time_format)
        df[['positive', 'neutral', 'negative']] = round(df[['positive', 'neutral', 'negative']], 2)
        return df[df['normalized_text'] != '']
