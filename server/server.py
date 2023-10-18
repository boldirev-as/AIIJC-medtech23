import json

import numpy as np
import pandas as pd
import torch

from flask import Flask, url_for, request
from flask_restful import Api

import matplotlib
import matplotlib.pyplot as plt

from inference import Lightning_ResNet1D, inference_model

matplotlib.use('TkAgg')

app = Flask(__name__)
api = Api(app)


@app.route('/api', methods=['POST'])
def api_request():
    request.files['file'].save("input.npy")

    preds = inference_model("input.npy", resnet_model)

    print(preds)
    return json.dumps({x: float(y) for x, y in zip(['перегородочный', 'передний', 'боковой',
                                                    'передне-боковой', 'передне-перегородочный', 'нижний', 'норма'],
                                                   preds)})


@app.route('/', methods=['POST', 'GET'])
def form_sample():
    if request.method == 'GET':
        return f'''<!doctype html>
                        <html lang="en">
                          <head>
                            <meta charset="utf-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
                            <link rel="stylesheet"
                            href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css"
                            integrity="sha384-giJF6kkoqNQ00vy+HMDP7azOuL0xtbfIcaT9wjKHr8RbDVddVHyTfAAsrekwKmP1"
                            crossorigin="anonymous">
                            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" 
                            integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" 
                            crossorigin="anonymous"></script>
                            <link rel="stylesheet" type="text/css" 
                                href="{url_for('static', filename='css/styles_for_select.css')}" />
                            <title>ЭКГ классификация</title>
                          </head>
                          <body>
                            <h1>Анкета пациента</h1>
                            <h2>для классификации и локализации инфаркта миокарда</h2>
                            <div>
                                <form method="POST" action="" enctype="multipart/form-data">
                                    <div class="input-group has-validation">
                                        <input type="text" class="form-control" id="age" 
                                            placeholder="Введите ваш возраст" name="age" required>
                                        <div class="invalid-feedback">
                                            Пожалуйста, введите Ваш возраст.
                                        </div>
                                    </div>
                                    <input type="text" class="form-control" 
                                        id="height" placeholder="Введите ваш рост" name="height">
                                    <input type="text" class="form-control" 
                                        id="height" placeholder="Введите ваш вес" name="weight">
                                    <div class="form-group">
                                        <label for="form-check">Укажите пол</label>
                                        <div class="form-check">
                                          <input class="form-check-input" type="radio" 
                                                name="sex" id="male" value="male" checked>
                                          <label class="form-check-label" for="male">
                                            Мужской
                                          </label>
                                        </div>
                                        <div class="form-check">
                                          <input class="form-check-input" type="radio" 
                                                name="sex" id="female" value="female">
                                          <label class="form-check-label" for="female">
                                            Женский
                                          </label>
                                        </div>
                                    </div>
                                    <div class="form-group">
                                        <label for="file">Приложите данные ЭКГ исследования</label>
                                        <input type="file" id="ecg_file" name="file" required>
                                    </div>
                                    <button type="submit" class="btn btn-primary">Отправить на анализ</button>
                                </form>
                            </div>
                          </body>
                        </html>'''
    elif request.method == 'POST':

        request.files['file'].save("input.npy")

        preds = inference_model("input.npy", resnet_model)

        ecg_signal = np.load("input.npy")
        pd.DataFrame(ecg_signal.T,
                     columns=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']).plot(
            subplots=True, figsize=(16, 8))
        plt.savefig("static/out.png", bbox_inches='tight')

        print(preds)
        return f"""<h3>Прогнозирование от модели: у вас инфаркт миокарда
         с вероятностью {round(100 - preds[6] * 100, 2)}%. <br>  
               Это может быть миокард: перегородочный ({round(preds[0] * 100, 2)}%), 
               передний ({round(preds[1] * 100, 2)}%), 
               боковой ({round(preds[2] * 100, 2)}%), передне-боковой ({round(preds[3] * 100, 2)}%),
                передне-перегородочный ({round(preds[4] * 100, 2)}%), нижний ({round(preds[5] * 100, 2)}%)</h3>
                <img src='{url_for('static', filename='out.png')}' class="img-fluid" alt="Responsive image">"""


if __name__ == '__main__':
    resnet_model = Lightning_ResNet1D.load_from_checkpoint("model.ckpt",
                                                           model_hparams={"n_classes": 7, "base_filters": 16,
                                                                          "kernel_size": 16, "stride": 2, "groups": 1,
                                                                          "n_block": 12, "in_channels": 12},
                                                           map_location=torch.device('cpu'))
    resnet_model.eval()

    app.run(port=8080, host='0.0.0.0')
