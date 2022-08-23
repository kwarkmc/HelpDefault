from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from chk import b_func, interpret
import time
app = Flask(__name__)

@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    if request.method == 'POST':
        try:
            f = request.files['file']
            f.save(secure_filename('file.jpg'))
            start = time.time()
            label = interpret("./file.jpg")
            end = time.time()
            duration = end - start
        except:
            return render_template('label0.html')

        if label == 1:
            return render_template('label1.html',data="{:.4f}".format(duration))
        elif label == 2:
            return render_template('label2.html',data="{:.4f}".format(duration))
        elif label == 3:
            return render_template('label3.html',data="{:.4f}".format(duration))
        elif label == 4:
            return render_template('label4.html',data="{:.4f}".format(duration))
        elif label == 5:
            return render_template('label5.html',data="{:.4f}".format(duration))
        elif label == 6:
            return render_template('label6.html',data="{:.4f}".format(duration))
        elif label == 7:
            return render_template('label7.html',data="{:.4f}".format(duration))
        elif label == 8:
            return render_template('label8.html',data="{:.4f}".format(duration))
        elif label == 9:
            return render_template('label9.html',data="{:.4f}".format(duration))
        elif label == 10:
            return render_template('label10.html',data="{:.4f}".format(duration))
        elif label == 11:
            return render_template('label11.html',data="{:.4f}".format(duration))
        else:
            return render_template('label0.html',data="{:.4f}".format(duration))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=8000)
