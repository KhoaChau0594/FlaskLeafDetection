from flask import Flask, render_template, jsonify, request, send_file, send_from_directory, make_response
import os
import utilis
import pickle
from utilis.classification import Classifier
import base64

# app = Flask(__name__)
# initialize classifier
# Load models
### create mask model
# path to model
create_mask_path = r'.\trained_model\create_ground_truth_model_5.pkl'
create_mask_model = pickle.load(open(create_mask_path, 'rb'))

### classified model
classify_model_path = r'.\trained_model\classification_model_3.pkl'
classify_model = pickle.load(open(classify_model_path, 'rb'))

classifier = Classifier(create_mask_model, classify_model)

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.jinja_env.auto_reload = True
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite')
    )

    app.config['TEMPLATES_AUTO_RELOAD'] = True

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # before request
    @app.before_request
    def before_request():
        # When you import jinja2 macros, they get cached which is annoying for local
        # development, so wipe the cache every request.
        if 'localhost' in request.host_url or '0.0.0.0' in request.host_url:
            app.jinja_env.cache = {}

    # router
    @app.route('/', methods=['GET', 'POST'])
    def hello():
        return render_template('homepage.html')

    @app.route('/_classify', methods=['GET', 'POST'])
    def add_numbers():
        file = request.files['image']
        dest = os.path.join(r'.\uploadedImage', file.filename)
        file.save(dest)

        print(dest)
        res = classifier.do_classify(dest)
        
        return jsonify({
            'result': res[0],
            'time': res[1]
        })
    
    @app.route('/get_segmented_image', methods=['GET', 'POST'])
    def get_segmented_image():
        imgName = request.form['imgName']
        segmentedPath = os.path.join(r'.\segmentedImage', imgName)
        # print('image name:', imgName)
        # return send_from_directory(
        #     r'.\segmentedImage',
        #     imgName, 
        #     as_attachment=True
        # )
        with open(segmentedPath, "rb") as f:
            image = f.read()

            response = make_response(base64.b64encode(image))
            response.headers.set('Content-Type', 'image/JPG')
            response.headers.set('Content-Disposition', 'attachment', filename='image.jpg')
            return response
        
        return None

    return app
