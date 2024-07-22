import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from collections import defaultdict

# Duplicate lib error fix
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

font_path = "Path to the font to use"
img_path = 'Path to source images'
result_path = 'Path to result outputs'
if not os.path.exists(result_path):
    os.makedirs(result_path)

imgNames = os.listdir(img_path)
imagePaths = [os.path.join(img_path, imgName) for imgName in imgNames]

ocr = PaddleOCR(use_angle_cls=True, lang='en')


def add_text_to_image(image, text, position, font_path, font_size=20):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill="red")
    return image

# preprocess functions
def preprocess_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def preprocess_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

def preprocess_adaptive_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

def preprocess_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def preprocess_median_blur(image, ksize=5):
    return cv2.medianBlur(image, ksize)

def preprocess_bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def preprocess_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def preprocess_sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def preprocess_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

def preprocess_hist_eq(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_eq_img = cv2.equalizeHist(gray)
    return cv2.cvtColor(hist_eq_img, cv2.COLOR_GRAY2BGR)

def preprocess_invert(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray)
    return cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)

def preprocess_erode(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_img = cv2.erode(image, kernel, iterations=1)
    return cv2.cvtColor(eroded_img, cv2.COLOR_GRAY2BGR) if len(eroded_img.shape) == 2 else eroded_img

def preprocess_dilate(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=1)
    return cv2.cvtColor(dilated_img, cv2.COLOR_GRAY2BGR) if len(dilated_img.shape) == 2 else dilated_img

def preprocess_opening(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    opened_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(opened_img, cv2.COLOR_GRAY2BGR) if len(opened_img.shape) == 2 else opened_img

def preprocess_closing(image, kernel_size=(3, 3)):
    kernel = np.ones(kernel_size, np.uint8)
    closed_img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.cvtColor(closed_img, cv2.COLOR_GRAY2BGR) if len(closed_img.shape) == 2 else closed_img

def preprocess_grayscale_erode_blur(image, erode_kernel_size=(3, 3), blur_kernel_size=(3, 3)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones(erode_kernel_size, np.uint8)
    eroded_img = cv2.erode(gray, kernel, iterations=1)
    blurred_img = cv2.GaussianBlur(eroded_img, blur_kernel_size, 0)
    return cv2.cvtColor(blurred_img, cv2.COLOR_GRAY2BGR)

def preprocess_hist_eq_erode_blur(image, erode_kernel_size=(3, 3), blur_kernel_size=(3, 3)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_eq_img = cv2.equalizeHist(gray)
    kernel = np.ones(erode_kernel_size, np.uint8)
    eroded_img = cv2.erode(hist_eq_img, kernel, iterations=1)
    blurred_img = cv2.GaussianBlur(eroded_img, blur_kernel_size, 0)
    return cv2.cvtColor(blurred_img, cv2.COLOR_GRAY2BGR)

def preprocess_gaussian_clahe_erode(image, blur_kernel_size=(5, 5), erode_kernel_size=(3, 3)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blurred_img)
    kernel = np.ones(erode_kernel_size, np.uint8)
    eroded_img = cv2.erode(clahe_img, kernel, iterations=1)
    return cv2.cvtColor(eroded_img, cv2.COLOR_GRAY2BGR)

def preprocess_gaussian_clahe(image, blur_kernel_size=(5, 5)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blurred_img)
    return cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

# Combine similar results with 0.80 conf, 3 recurrence
def combine_similar_results(results):
    filtered_results = [res for res in results if res['confidence'] > 0.80]
    
    text_scores = defaultdict(list)
    
    # Aggregate scores for each text
    for res in filtered_results:
        text_scores[res['text']].append(res['confidence'])
    
    combined_results = []
    for text, scores in text_scores.items():
        if len(scores) > 3:
            average_confidence = sum(scores) / len(scores)
            combined_results.append({
                "text": text,
                "confidence": average_confidence
            })
    
    # Sort results by confidence
    combined_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return combined_results

# Preprocess loop
for img in imagePaths:
    try:
        image = cv2.imread(img)
        if image is None:
            print(f"Error reading image {img}")
            continue
        
        img_name = os.path.basename(img).split('.')[0]
        all_results = []

        preprocess_funcs = {
            "Original": lambda x: x,
            "Grayscale": preprocess_grayscale,
            "Threshold": preprocess_threshold,
            "Adaptive Threshold": preprocess_adaptive_threshold,
            "Gaussian Blur 5x5": lambda x: preprocess_blur(x, (5, 5)),
            "Gaussian Blur 9x9": lambda x: preprocess_blur(x, (9, 9)),
            "Median Blur": preprocess_median_blur,
            "Bilateral Filter": preprocess_bilateral_filter,
            "Sharpen": preprocess_sharpen,
            "CLAHE": preprocess_clahe,
            "Histogram Equalization": preprocess_hist_eq,
            "Invert": preprocess_invert,
            "Erode 3x3": lambda x: preprocess_erode(x, (3, 3)),
            "Dilate 3x3": lambda x: preprocess_dilate(x, (3, 3)),
            "Opening 3x3": lambda x: preprocess_opening(x, (3, 3)),
            "Closing 3x3": lambda x: preprocess_closing(x, (3, 3)),
            # "Grayscale + Invert": lambda x: cv2.cvtColor(preprocess_invert(preprocess_grayscale(x), cv2.COLOR_GRAY2BGR)),
            "Grayscale + Blur": lambda x: preprocess_blur(preprocess_grayscale(x), (5, 5)),
            "Gaussian + CLAHE": lambda x: preprocess_gaussian_clahe(x, (5, 5)),
            # "Grayscale + Edges": lambda x: preprocess_edges(cv2.cvtColor(preprocess_grayscale(x), cv2.COLOR_GRAY2BGR)),
            "Gaussian + CLAHE + Erode 3x3": lambda x: preprocess_gaussian_clahe_erode(x, (5, 5), (3, 3)),
            "Grayscale + Erode 3x3 + Gaussian Blur 3x3": lambda x: preprocess_grayscale_erode_blur(x, (3, 3), (3, 3)),
            "Grayscale + Erode 3x3 + Gaussian Blur 5x5": lambda x: preprocess_grayscale_erode_blur(x, (3, 3), (5, 5)),
            "Hist Eq + Erode 3x3 + Gaussian Blur 3x3": lambda x: preprocess_hist_eq_erode_blur(x, (3, 3), (3, 3)),
        }

        fig, axes = plt.subplots(5, 5, figsize=(70, 70))
        axes = axes.flatten()
        
        #Process every method generated image with paddleOCR
        for ax, (method, func) in zip(axes, preprocess_funcs.items()):
            try:
                processed_img = func(image)
                
                if len(processed_img.shape) == 2:
                    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                
                #OCR#
                ocr_result = ocr.ocr(processed_img, cls=False)
                #OCR#


                #Append results to all_results to combine them
                for res in ocr_result:
                    for line in res:
                        all_results.append({
                            "text": line[1][0],
                            "confidence": line[1][1]
                        })

                if ocr_result:
                    imageWithRect = Image.fromarray(processed_img)
                    boxes = [line[0] for res in ocr_result for line in res]
                    txts = [line[1][0] for res in ocr_result for line in res]
                    scores = [line[1][1] for res in ocr_result for line in res]
                    im_show = draw_ocr(np.array(imageWithRect), boxes, txts, scores, font_path)
                    im_show = Image.fromarray(im_show)
                    im_show = add_text_to_image(im_show, method, (10, 10), font_path)
                    ax.imshow(im_show)
                    ax.set_title(method)
                else:
                    ax.imshow(processed_img, cmap='gray')
                    ax.set_title(f"{method} - No text found")
            except Exception as e:
                print(f"Error processing method {method} for image {img}: {e}")
                ax.set_title(f"{method} - Error")
        
        plt.tight_layout()
        plt.savefig(os.path.join(result_path, f"{img_name}-combined-result.png"))
        plt.show()

        # Combine and save results
        combined_results = combine_similar_results(all_results)
        combined_img = Image.fromarray(image)
        for idx, result in enumerate(combined_results):
            combined_img = add_text_to_image(combined_img, f"{result['text']} - {result['confidence']:.3f}", (10, 10 + 30 * idx), font_path)
        
        combined_img.save(os.path.join(result_path, f"{img_name}-combined-results.png"))
        plt.imshow(combined_img)
        plt.title(f"Results for {img_name}")
        plt.show()

    except Exception as e:
        print(f"Error processing image {img}: {e}")
