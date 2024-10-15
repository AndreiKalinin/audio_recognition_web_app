from flask import Flask, render_template, request, send_file, abort
import dotenv
import os
import logging
import hashlib
import io
from time import sleep
from datetime import datetime
import pandas as pd
from waitress import serve
from audio_transcription.transcribe_audio import AudioTranscriber


dotenv.load_dotenv()
pass_hash = os.getenv('PASS_HASH')
audio_token = os.getenv('AUDIO_TOKEN')

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        password = request.form.get('password')
        link = request.form.get('text')

        if hashlib.md5(password.encode('utf-8')).hexdigest() != pass_hash:
            return "Неправильный пароль", 403

        transcriber = AudioTranscriber(audio_token, 'tls_certificate/certificate.cer')
        transcriber.get_token()
        logger.info('Token has been created')
        request_id, num_channels, sample_rate, file_length = transcriber.upload_file(link)
        logger.info('File has been uploaded')
        task_id = transcriber.send_transcription_task(request_id, sample_rate, num_channels)
        logger.info('Task has been sent')
        delay_time = file_length / 10
        output_df_buffer = io.BytesIO()
        for i in range(3):
            sleep(delay_time)
            task_state = transcriber.check_status(task_id)
            if task_state['status'] == 'DONE':
                logger.info('File has been successfully transcribed into text')
                df_result = transcriber.download_result(task_state['response_file_id'])
                logger.info('Result has been received')
                with pd.ExcelWriter(output_df_buffer, engine='xlsxwriter') as writer:
                    df_result.to_excel(writer, index=False, sheet_name='results')
                    workbook = writer.book
                    worksheet = writer.sheets['results']
                    wrap_format = workbook.add_format({'text_wrap': True})
                    worksheet.set_column('A:A', 100, wrap_format)
                logger.info('Result has been saved into file')
                filename = 'output_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.xlsx'
                output_df_buffer.seek(0)
                logger.info('Document has been sent via bot')
                return send_file(output_df_buffer,
                          as_attachment=True,
                          download_name=filename,
                          mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        output_df_buffer.close()
    return render_template('submit.html')


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
