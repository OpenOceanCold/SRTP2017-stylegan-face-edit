
from flask import Flask,render_template,request
import os
import sys
from datetime import timedelta
import time


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

@app.route('/')
def main():
    return render_template('Stylegan.html')

@app.route('/src_get')
def src_get():
    src_path = str(request.args['srcpath'])
    cmd = "python Face_detect.py " + src_path
    os.system(cmd)
    face_path = "/static/" + os.path.basename(os.path.splitext(src_path)[0]) + '_face.png'
    print(face_path)
    return render_template('Stylegan.html', face_detected=face_path)

@app.route('/face_encode')
def face_encode():
    face_path = str(request.args['facepath'])
    num = str(request.args['network'])
    cmd = "python Face_encode.py " + num + ' ' + face_path
    os.system(cmd)
    generated_path = "/static/" + os.path.basename(os.path.splitext(face_path)[0]) + '_generated.png'
    return render_template('Stylegan.html',face_generated=generated_path)

@app.route('/face_edit')
def face_edit():
    npy_path = str(request.args['npypath'])
    num = str(request.args['network2'])
    beauty = str(request.args['beauty'])
    gender = str(request.args['gender'])
    height = str(request.args['height'])
    width = str(request.args['width'])
    age = str(request.args['age'])
    horizontal = str(request.args['horizontal'])
    vertical = str(request.args['vertical'])

    cmd = "python Face_edit.py " + num + ' ' + npy_path + ' ' + beauty + ' ' + gender + ' ' + height + ' ' + width + ' ' + age + ' ' + horizontal + ' ' + vertical

    os.system(cmd)
    edited_path = "/static/" + os.path.basename(os.path.splitext(npy_path)[0])+ '_Edited.png'
    return render_template('Stylegan.html',face_edited=edited_path)


    
if __name__ == '__main__':
    app.run(debug=True, port=8080)
