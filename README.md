# README  
## Задание 1: Программирование (interview_1.ipynb)

*    Написал 4 различных решения
*    Создал тестовые массивы для проверки
*    Проверил дают ли решения одинаковые правильные ответы
*    Замерил скорость выполнения каждого из решений
*    Выбрал наиболее читаемое и быстрое решение  
[Ссылка interview_1.ipynb](https://github.com/AIntosha/interview/blob/master/interview_1.ipynb)  
  
## Задание 2: Gender recognition
Решил задачу распознания пола человека по фотографии лица двумя разными способами:  
* С помощью fastai ([interview_2_fastai.ipynb](https://github.com/AIntosha/interview/blob/master/interview_2_fastai.ipynb))  
14 эпох, accuracy **0.9824**    
[Ссылка на скачивание готовой модели export.pkl](https://cloud.mail.ru/public/7cbL/32iu3z8re)  
[Альтернативная ссылка](https://drive.google.com/file/d/15gx30Emh2pEGLVUZPA9rs1f3jRMnWoAd/view?usp=sharing)  
  
     
* С помощью pytorch ([interview_2_pytorch.ipynb](https://github.com/AIntosha/interview/blob/master/interview_2_pytorch.ipynb))  
100 эпох, accuracy **0.9542**    
[Ссылка на скачивание готовой модели model_pytorch.pth.tar](https://cloud.mail.ru/public/3W1W/jzvMR4ryr)  
[Альтернативная ссылка](https://drive.google.com/file/d/1bG2FBwSJBZSwj-ILWG_tO3dMZwOMB94L/view?usp=sharing)
  
#### Для воспроизведения результатов использовались:
* ubuntu 18
* виртуальное окружение conda - python 3.7
* библиотеки из [requirements.txt](https://github.com/AIntosha/interview/blob/master/requirements.txt)  
* модели должны лежать в корневой папке проекта
  
#### Как проверить на своем датасете:
* Модель fastai  
`python3 check_fastai.py /folder/to/files`  
  
* Модель pytorch  
`python3 check_pytorch.py /folder/to/files`
  
где */folder/to/files* - путь до вашей папки с изображениями  
  
По завершении, в корневой папке будет создан файл `process_results.json`  
с содержанием вида   
`{
    "image1.png": "male",
    "image2.jpg": "female",
}`
