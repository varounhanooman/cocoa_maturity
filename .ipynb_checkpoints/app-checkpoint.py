import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import cv2
from matplotlib import pyplot as plt
from scipy.stats import norm

app = Flask(__name__)
model = pickle.load(open('finalized_model_logreg.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    def upload_file():
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
            return uploaded_file.filename
        return redirect(url_for('home'))

    def Process_image(image_name):
        image_vec = cv2.imread(image_name, 1)
        g_blurred = cv2.GaussianBlur(image_vec, (5, 5), 0)

        blurred_float = g_blurred.astype(np.float32) / 255.0
        edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
        edges = edgeDetector.detectEdges(blurred_float) * 255.0
        cv2.imwrite('edge-raw.jpg', edges)

        def SaltPepperNoise(edgeImg):

            count = 0
            lastMedian = edgeImg
            median = cv2.medianBlur(edgeImg, 3)
            while not np.array_equal(lastMedian, median):
                zeroed = np.invert(np.logical_and(median, edgeImg))
                edgeImg[zeroed] = 0
                count = count + 1
                if count > 70:
                    break
                lastMedian = median
                median = cv2.medianBlur(edgeImg, 3)
        edges_ = np.asarray(edges, np.uint8)
        SaltPepperNoise(edges_)
        cv2.imwrite('edge.jpg', edges_)
        #image_display('edge.jpg')

        def findSignificantContour(edgeImg):
            contours, hierarchy = cv2.findContours(
                edgeImg,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
                # Find level 1 contours
            level1Meta = []
            for contourIndex, tupl in enumerate(hierarchy[0]):
                # Filter the ones without parent
                if tupl[3] == -1:
                    tupl = np.insert(tupl.copy(), 0, [contourIndex])
                    level1Meta.append(tupl)
        # From among them, find the contours with large surface area.
            contoursWithArea = []
            for tupl in level1Meta:
                contourIndex = tupl[0]
                contour = contours[contourIndex]
                area = cv2.contourArea(contour)
                contoursWithArea.append([contour, area, contourIndex])
            contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
            largestContour = contoursWithArea[0][0]
            return largestContour
        contour = findSignificantContour(edges_)
        # Draw the contour on the original image
        contourImg = np.copy(image_vec)
        cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
        cv2.imwrite('contour.jpg', contourImg)
        #image_display('contour.jpg')

        mask = np.zeros_like(edges_)
        cv2.fillPoly(mask, [contour], 255)
        # calculate sure foreground area by dilating the mask
        mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)
        # mark inital mask as "probably background"
        # and mapFg as sure foreground
        trimap = np.copy(mask)
        trimap[mask == 0] = cv2.GC_BGD
        trimap[mask == 255] = cv2.GC_PR_BGD
        trimap[mapFg == 255] = cv2.GC_FGD
        # visualize trimap
        trimap_print = np.copy(trimap)
        trimap_print[trimap_print == cv2.GC_PR_BGD] = 255
        trimap_print[trimap_print == cv2.GC_FGD] = 255
        cv2.imwrite('trimap.png', trimap_print)
        #image_display('trimap.png')

        mask_test = cv2.imread('trimap.png')/255.0
        final = (image_vec * mask_test).clip(0, 255).astype(np.uint8)
        cv2.imwrite('final.png', final)

        #extract red channel
        red_channel = np.array(final[:,:,2])
        red_channel = red_channel.flatten()
        red_channel = red_channel[red_channel!=0]
        rmu, rstd = norm.fit(red_channel)

        #extract green channel
        green_channel = np.array(final[:,:,1])
        green_channel = green_channel.flatten()
        green_channel = green_channel[green_channel!=0]
        gmu, gstd = norm.fit(green_channel)

        #extract blue channel
        blue_channel = np.array(final[:,:,0])
        blue_channel = blue_channel.flatten()
        blue_channel = blue_channel[blue_channel!=0]
        bmu, bstd = norm.fit(blue_channel)


        result = [rmu, gmu, bmu, rstd, gstd, bstd]
        #result = [202, 123, 37, 10, 9, 8]
        return result

    image_name = upload_file()
    Process_result = Process_image(image_name)
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    final_features = [np.array(Process_result)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Prediction {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    
