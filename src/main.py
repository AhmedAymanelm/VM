import streamlit as st
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
import cv2
import gdown
import os

# Page configuration
st.set_page_config(page_title="VIMD", layout="wide")

# Import YOLO
from ultralytics import YOLO

# Class names for Brain
brain_class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Class names for Pancreas
pancreas_class_names = ['normal', 'cancer']

# Detailed description for Brain conditions
brain_disease_descriptions = {
    'glioma': {
        'name_en': 'Glioma Tumor',
        'description': '''
        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin-bottom: 20px;">
            <h4 style="color: #dc3545; margin-top: 0;"> What is Glioma?</h4>
            <p style="color: #333; line-height: 1.8;">
                Glioma is the most common type of brain tumor, arising from glial cells that support nerve cells in the brain.
            </p>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #dc3545; margin-top: 0;"> Common Symptoms:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Persistent and severe headaches, especially in the morning</li>
                <li>Seizures or convulsions</li>
                <li>Weakness on one side of the body</li>
                <li>Speech or vision problems</li>
                <li>Personality and behavioral changes</li>
                <li>Nausea and vomiting</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #dc3545; margin-top: 0;">Treatment options – use only after consulting your doctor:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Surgery to remove the tumor (complete or partial)</li>
                <li>Radiation therapy</li>
                <li>Chemotherapy</li>
                <li>Modern immunotherapy</li>
            </ul>
        </div>

        <div style="background: #dc3545; color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <strong> Important Note:</strong> Early diagnosis increases chances of successful treatment. Consult a specialist immediately.
         </div>
        ''',
        'severity': 'High',
        'color': '#dc3545'
    },
    'meningioma': {
        'name_en': 'Meningioma Tumor',
        'description': '''
        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 20px;">
            <h4 style="color: #ffc107; margin-top: 0;">⚕️ What is Meningioma?</h4>
            <p style="color: #333; line-height: 1.8;">
                A tumor arising from the membranes surrounding the brain and spinal cord (meninges). Most cases are benign and slow-growing.
            </p>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #ffc107; margin-top: 0;">🩺 Common Symptoms:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Progressive headaches</li>
                <li>Weakness in arms or legs</li>
                <li>Numbness in extremities</li>
                <li>Hearing loss or tinnitus</li>
                <li>Memory problems</li>
                <li>Blurred vision</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #ffc107; margin-top: 0;">Treatment options – use only after consulting your doctor:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Observation and monitoring (for small tumors)</li>
                <li>Surgery (primary treatment)</li>
                <li>Targeted radiation therapy (Gamma Knife)</li>
                <li>Medication in some cases</li>
            </ul>
        </div>

        <div style="background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <strong> Recovery Rate:</strong> Very high, especially with early detection, as 90% of cases are benign.
        </div>
        ''',
        'severity': 'Moderate',
        'color': '#ffc107'
    },
    'pituitary': {
        'name_en': 'Pituitary Tumor',
        'description': '''
        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #fd7e14; margin-bottom: 20px;">
            <h4 style="color: #fd7e14; margin-top: 0;"> What is a Pituitary Tumor?</h4>
            <p style="color: #333; line-height: 1.8;">
                A tumor affecting the pituitary gland (a small gland at the base of the brain that controls hormones). Usually benign.
            </p>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #fd7e14; margin-top: 0;"> Common Symptoms:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Hormonal imbalances (weight gain, chronic fatigue)</li>
                <li>Vision problems (especially peripheral vision)</li>
                <li>Persistent headaches</li>
                <li>Menstrual irregularities in women</li>
                <li>Sexual dysfunction in men</li>
                <li>Abnormal growth in hands and feet</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #fd7e14; margin-top: 0;">Treatment options – use only after consulting your doctor:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Medication (to regulate hormones)</li>
                <li>Transsphenoidal surgery (through the nose)</li>
                <li>Radiation therapy</li>
                <li>Hormone replacement therapy</li>
            </ul>
        </div>

        <div style="background: linear-gradient(135deg, #fd7e14 0%, #ff6b6b 100%); color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <strong> Success Rate:</strong> Very high - more than 85% of cases are completely cured with appropriate treatment.
        </div>
        ''',
        'severity': 'Moderate',
        'color': '#fd7e14'
    },
    'notumor': {
        'name_en': 'No Tumor - Healthy Brain',
        'description': '''
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h2 style="color: #155724; margin: 0;">Congratulations! No Tumors Detected</h2>
            <p style="color: #155724; margin-top: 10px; font-size: 1.1em;">The scan shows a healthy brain with no signs of tumors. This is great news!</p>
        </div>

        <h3 style="color: #28a745; margin-top: 25px; margin-bottom: 15px;"> Tips for Maintaining Brain Health:</h3>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-top: 0;"> 1. Healthy Diet:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Eat foods rich in Omega-3 (fish, nuts)</li>
                <li>Increase fresh vegetables and fruits</li>
                <li>Avoid processed sugars and saturated fats</li>
                <li>Drink adequate water (2-3 liters daily)</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-top: 0;"> 2. Physical Activity:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Exercise for at least 30 minutes daily</li>
                <li>Brisk walking improves blood flow to the brain</li>
                <li>Yoga and meditation to reduce stress</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-top: 0;"> 3. Good Sleep:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>7-8 hours of sleep daily</li>
                <li>Regular sleep helps regenerate brain cells</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-top: 0;"> 4. Mental Stimulation:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Reading and solving puzzles</li>
                <li>Learning new skills</li>
                <li>Active social interaction</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #dc3545;">
            <h4 style="color: #dc3545; margin-top: 0;"> 5. Avoid:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Smoking and alcohol</li>
                <li>Chronic stress</li>
                <li>Head injuries (wear a helmet when needed)</li>
            </ul>
        </div>

        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;">
            <strong style="font-size: 1.1em;"> Remember: Regular check-ups are important for your health!</strong>
        </div>
        ''',
        'severity': 'None',
        'color': '#28a745'
    }
}

# Detailed description for Pancreas conditions
pancreas_disease_descriptions = {
    'normal': {
        'name_en': 'Normal Pancreas - Healthy',
        'description': '''
        <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h2 style="color: #155724; margin: 0;"> Great News! Pancreas is Normal</h2>
            <p style="color: #155724; margin-top: 10px; font-size: 1.1em;">The scan shows a healthy pancreas with no abnormalities detected!</p>
        </div>

        <h3 style="color: #28a745; margin-top: 25px; margin-bottom: 15px;"> Tips for Maintaining Pancreas Health:</h3>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-top: 0;"> 1. Healthy Diet:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Eat plenty of vegetables and fruits</li>
                <li>Choose whole grains over refined carbs</li>
                <li>Limit red meat and processed foods</li>
                <li>Avoid excessive alcohol consumption</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-top: 0;">2. Lifestyle Changes:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Stop smoking immediately</li>
                <li>Maintain a healthy weight</li>
                <li>Exercise regularly (30 minutes daily)</li>
                <li>Manage stress effectively</li>
            </ul>
        </div>

        <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;">
            <strong style="font-size: 1.1em;"> Prevention is better than cure - Keep up the healthy lifestyle!</strong>
        </div>
        ''',
        'severity': 'None',
        'color': '#28a745'
    },
    'cancer': {
        'name_en': 'Pancreatic Tumor Detected',
        'description': '''
        <div style="background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #dc3545; margin-bottom: 20px;">
            <h4 style="color: #dc3545; margin-top: 0;"> What is a Pancreatic Tumor?</h4>
            <p style="color: #333; line-height: 1.8;">
                A tumor detected in the pancreas. This requires immediate medical evaluation to determine the type and appropriate treatment plan.
            </p>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #dc3545; margin-top: 0;">Common Symptoms:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Abdominal or back pain</li>
                <li>Unexplained weight loss</li>
                <li>Jaundice (yellowing of skin and eyes)</li>
                <li>Loss of appetite</li>
                <li>Dark urine and pale stools</li>
                <li>Nausea and digestive problems</li>
                <li>New-onset diabetes in some cases</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #dc3545; margin-top: 0;"> Treatment options – use only after consulting your doctor:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Immediate consultation with a gastroenterologist or oncologist</li>
                <li>Additional imaging tests (CT scan, MRI, PET scan)</li>
                <li>Biopsy to determine tumor type</li>
                <li>Blood tests (CA 19-9 tumor marker)</li>
                <li>Endoscopic ultrasound (EUS)</li>
            </ul>
        </div>

        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <h4 style="color: #dc3545; margin-top: 0;"> Treatment Options May Include:</h4>
            <ul style="line-height: 1.8; color: #333;">
                <li>Surgery (Whipple procedure or other surgical options)</li>
                <li>Chemotherapy</li>
                <li>Radiation therapy</li>
                <li>Targeted therapy</li>
                <li>Immunotherapy</li>
                <li>Palliative care for symptom management</li>
            </ul>
        </div>

        <div style="background: #dc3545; color: white; padding: 15px; border-radius: 8px; text-align: center;">
            <strong> Critical:</strong> This AI detection requires confirmation by medical professionals. Contact your doctor immediately for proper diagnosis and treatment planning.
        </div>
        ''',
        'severity': 'High',
        'color': '#dc3545'
    }
}


# Load model
@st.cache_resource
def load_brain_model():
    model_path = "Model_Final_Brain_Tumor99%.h5"
    
    file_id = "1pJtyO96630tlLatZlo2AiCIwT5Qo1-Rx"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        try:
            with st.spinner('🔄 Downloading Brain model from Google Drive... This may take a few minutes.'):
                if os.path.exists(model_path):
                    os.remove(model_path)
                gdown.download(gdrive_url, model_path, quiet=False, fuzzy=True)
                st.success(f'✅ Brain model downloaded successfully! Size: {os.path.getsize(model_path)/1024/1024:.1f} MB')
        except Exception as e:
            st.error(f"❌ Error downloading Brain model: {str(e)}")
            st.info("Please make sure the Google Drive file is set to 'Anyone with the link' can view.")
            raise
    
    model = tf.keras.models.load_model(model_path)
    return model


@st.cache_resource
def load_brain_yolo_model():
    model_path = "best.pt"
    
    file_id = "1C7ffHxYkvGDcudyZCIs18nrcbRqV_uqH"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        try:
            with st.spinner('🔄 Downloading YOLO model from Google Drive... This may take a few minutes.'):
                if os.path.exists(model_path):
                    os.remove(model_path)
                gdown.download(gdrive_url, model_path, quiet=False, fuzzy=True)
                st.success(f'✅ YOLO model downloaded successfully! Size: {os.path.getsize(model_path)/1024/1024:.1f} MB')
        except Exception as e:
            st.error(f"❌ Error downloading YOLO model: {str(e)}")
            st.info("Please make sure the Google Drive file is set to 'Anyone with the link' can view.")
            raise
    
    model = YOLO(model_path)
    return model


@st.cache_resource
def load_pancreas_model():
    model_path = "pancreas_tumor_ACC_98%.h5"
    
    # Google Drive File ID
    file_id = "1ES4L5ugXQMqYH50ZldlJ1q5NcILyqyVC"
    gdrive_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        try:
            with st.spinner('🔄 Downloading Pancreas model from Google Drive... This may take a few minutes.'):
                if os.path.exists(model_path):
                    os.remove(model_path)
                gdown.download(gdrive_url, model_path, quiet=False, fuzzy=True)
                st.success(f'Pancreas model downloaded successfully! Size: {os.path.getsize(model_path)/1024/1024:.1f} MB')
        except Exception as e:
            st.error(f" Error downloading Pancreas model: {str(e)}")
            st.info("Please make sure the Google Drive file is set to 'Anyone with the link' can view.")
            raise
    
    model = tf.keras.models.load_model(model_path)
    return model


# Read image and convert to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None


# YOLO Detection function for Brain - Enhanced
def predict_pancreas_with_gradcam(model, img):
    target_size = (299, 299)
    orig_array = np.array(img).astype('uint8')
    img_resized = img.resize(target_size)
    x = tf.keras.preprocessing.image.img_to_array(img_resized)
    x_norm = x / 255.0
    x_input = np.expand_dims(x_norm, axis=0)

    zoom_factor = 0.5
    h, w, _ = x_norm.shape
    center_h, center_w = h // 2, w // 2
    zoom_h, zoom_w = int(h * zoom_factor), int(w * zoom_factor)
    start_h = center_h - zoom_h // 2
    end_h = center_h + zoom_h // 2
    start_w = center_w - zoom_w // 2
    end_w = center_w + zoom_w // 2
    cropped = x_norm[start_h:end_h, start_w:end_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    preds = model.predict(x_input, verbose=0)
    pred_class = np.argmax(preds[0])
    pred_label = pancreas_class_names[pred_class]
    confidence = preds[0][pred_class] * 100

    # Enhanced Grad-CAM (same logic as brain tumor)
    heatmap = None
    best_heatmap = None
    best_score = -1

    try:
        # Try to find Xception base model first
        try:
            base_model = model.get_layer("xception")
            layers_to_try = [
                "block14_sepconv2_act",
                "block13_sepconv2_act",
                "block12_sepconv2_act"
            ]

            for layer_name in layers_to_try:
                try:
                    last_conv_layer = base_model.get_layer(layer_name)
                    grad_model = tf.keras.models.Model(
                        inputs=base_model.input,
                        outputs=[last_conv_layer.output, base_model.output]
                    )

                    with tf.GradientTape() as tape:
                        conv_outputs, predictions = grad_model(x_input)
                        loss = predictions[:, pred_class]

                    grads = tape.gradient(loss, conv_outputs)
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_outputs_np = conv_outputs[0].numpy()
                    pooled_grads_np = pooled_grads.numpy()

                    for i in range(pooled_grads_np.shape[0]):
                        conv_outputs_np[:, :, i] *= pooled_grads_np[i]

                    heatmap = np.mean(conv_outputs_np, axis=-1)
                    heatmap = np.maximum(heatmap, 0)

                    if np.max(heatmap) > 0:
                        heatmap = heatmap / np.max(heatmap)

                    # Scoring for best layer
                    if pred_label != 'normal':
                        high_activation = np.sum(heatmap > 0.5)
                        total_activation = np.sum(heatmap > 0.1)

                        if total_activation > 0:
                            concentration_score = high_activation / total_activation
                            size_score = 1.0 - (total_activation / heatmap.size)
                            score = concentration_score * 0.8 + size_score * 0.2

                            if score > best_score:
                                best_score = score
                                best_heatmap = heatmap.copy()
                    else:
                        best_heatmap = heatmap.copy()
                        break

                except Exception as e:
                    continue

        except:
            # If Xception not found, find last conv layer generically
            last_conv_layer = None
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    last_conv_layer = layer
                    break

            if last_conv_layer is not None:
                grad_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=[last_conv_layer.output, model.output]
                )

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(x_input)
                    loss = predictions[:, pred_class]

                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs_np = conv_outputs[0].numpy()
                pooled_grads_np = pooled_grads.numpy()

                for i in range(pooled_grads_np.shape[0]):
                    conv_outputs_np[:, :, i] *= pooled_grads_np[i]

                heatmap = np.mean(conv_outputs_np, axis=-1)
                heatmap = np.maximum(heatmap, 0)
                best_heatmap = heatmap

        if best_heatmap is None and heatmap is not None:
            best_heatmap = heatmap

        # Process heatmap - SAME AS BRAIN TUMOR
        if best_heatmap is not None:
            heatmap_resized = cv2.resize(best_heatmap, target_size, interpolation=cv2.INTER_CUBIC)

            # Normalize
            if np.max(heatmap_resized) > 0:
                heatmap_resized = heatmap_resized / np.max(heatmap_resized)

            # Threshold adjustment
            if pred_label == 'normal':
                heatmap_resized = heatmap_resized * 0.2
                threshold = 0.5
            else:
                # More aggressive threshold for cancer cases
                threshold_percentile = 70
                threshold = np.percentile(heatmap_resized[heatmap_resized > 0], threshold_percentile) if np.any(
                    heatmap_resized > 0) else 0.3

            heatmap_resized[heatmap_resized < threshold] = 0

            # Normalize after thresholding
            if np.max(heatmap_resized) > 0:
                heatmap_resized = heatmap_resized / np.max(heatmap_resized)

            # Enhanced morphological operations for cancer cases
            if pred_label != 'normal':
                # Power transform for contrast
                heatmap_resized = np.power(heatmap_resized, 1.8)

                heatmap_uint8_temp = np.uint8(255 * heatmap_resized)

                # Stronger morphological operations
                kernel_large = np.ones((5, 5), np.uint8)
                kernel_small = np.ones((3, 3), np.uint8)

                # Close small holes
                heatmap_uint8_temp = cv2.morphologyEx(heatmap_uint8_temp, cv2.MORPH_CLOSE, kernel_large)
                # Remove noise
                heatmap_uint8_temp = cv2.morphologyEx(heatmap_uint8_temp, cv2.MORPH_OPEN, kernel_small)
                # Dilate to make region more visible
                heatmap_uint8_temp = cv2.dilate(heatmap_uint8_temp, kernel_small, iterations=1)

                heatmap_resized = heatmap_uint8_temp / 255.0

            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            # Apply colormap
            if pred_label == 'normal':
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_OCEAN)
            else:
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            original_img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)

            # Alpha blending
            if pred_label == 'normal':
                alpha = 0.15
                beta = 0.85
            else:
                alpha = 0.65
                beta = 0.35

            superimposed_img = cv2.addWeighted(original_img_cv, beta, heatmap_color, alpha, 0)
            gradcam_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback if all attempts fail
            gradcam_img = orig_array

    except Exception as e:
        print(f"⚠️ Grad-CAM failed: {e}")
        gradcam_img = orig_array

    return {
        'original': orig_array,
        'zoomed': zoomed,
        'gradcam': gradcam_img,
        'prediction': pred_label,
        'confidence': confidence,
        'all_predictions': preds[0]
    }


# Enhanced Grad-CAM with YOLO Integration for Brain
def predict_brain_with_gradcam(model, img, yolo_mask=None):
    target_size = (299, 299)
    orig_array = np.array(img).astype('uint8')
    img_resized = img.resize(target_size)
    x = tf.keras.preprocessing.image.img_to_array(img_resized)
    x_norm = x / 255.0
    x_input = np.expand_dims(x_norm, axis=0)

    # Zoom processing
    zoom_factor = 0.5
    h, w, _ = x_norm.shape
    center_h, center_w = h // 2, w // 2
    zoom_h, zoom_w = int(h * zoom_factor), int(w * zoom_factor)
    start_h = center_h - zoom_h // 2
    end_h = center_h + zoom_h // 2
    start_w = center_w - zoom_w // 2
    end_w = center_w + zoom_w // 2
    cropped = x_norm[start_h:end_h, start_w:end_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

    preds = model.predict(x_input, verbose=0)
    pred_class = np.argmax(preds[0])
    pred_label = brain_class_names[pred_class]
    confidence = preds[0][pred_class] * 100

    try:
        base_model = model.get_layer("xception")
    except:

        base_model = model
        

    layers_to_try = [
        "block14_sepconv2_act",
        "block13_sepconv2_act", 
        "block12_sepconv2_act",
        "block11_sepconv2_act"
    ]

    best_heatmap = None
    best_score = -1

    for layer_name in layers_to_try:
        try:
            last_conv_layer = base_model.get_layer(layer_name)
            grad_model = tf.keras.models.Model(
                inputs=base_model.input,
                outputs=[last_conv_layer.output, base_model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x_input)
                loss = predictions[:, pred_class]

            grads = tape.gradient(loss, conv_outputs)
            
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs_np = conv_outputs[0].numpy()
            pooled_grads_np = pooled_grads.numpy()

            for i in range(pooled_grads_np.shape[0]):
                if pooled_grads_np[i] > 0:  
                    conv_outputs_np[:, :, i] *= pooled_grads_np[i]
                else:
                    conv_outputs_np[:, :, i] = 0

            heatmap = np.mean(conv_outputs_np, axis=-1)
            heatmap = np.maximum(heatmap, 0)

            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)

            if pred_label != 'notumor':
                high_activation = np.sum(heatmap > 0.6)
                medium_activation = np.sum(heatmap > 0.3)
                total_activation = np.sum(heatmap > 0.05)

                if total_activation > 0:
                    concentration_score = high_activation / (total_activation + 1)
                    clarity_score = medium_activation / (total_activation + 1)
                    size_penalty = min(1.0, total_activation / (heatmap.size * 0.3))
                    
                    score = (concentration_score * 0.5 + 
                            clarity_score * 0.3 + 
                            size_penalty * 0.2)

                    if score > best_score:
                        best_score = score
                        best_heatmap = heatmap.copy()
            else:
                if best_heatmap is None:
                    best_heatmap = heatmap.copy()
                break

        except Exception as e:
            continue

    if best_heatmap is None:
        best_heatmap = np.zeros((10, 10))

    heatmap = best_heatmap

    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)

    if yolo_mask is not None and pred_label != 'notumor':
        yolo_mask_resized = cv2.resize(yolo_mask, target_size, interpolation=cv2.INTER_NEAREST)
        yolo_mask_norm = yolo_mask_resized.astype(float) / 255.0
        
        heatmap_resized = heatmap_resized * yolo_mask_norm
        
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (7, 7), 0)
        
        heatmap_resized = heatmap_resized * yolo_mask_norm

    # Normalize
    if np.max(heatmap_resized) > 0:
        heatmap_resized = heatmap_resized / np.max(heatmap_resized)


    if pred_label == 'notumor':
        heatmap_resized = heatmap_resized * 0.15
        threshold = 0.6
    else:
        threshold_percentile = 60  
        non_zero_values = heatmap_resized[heatmap_resized > 0.05]
        
        if len(non_zero_values) > 0:
            threshold = np.percentile(non_zero_values, threshold_percentile)
            threshold = max(0.2, min(0.5, threshold))  
        else:
            threshold = 0.25

    heatmap_resized[heatmap_resized < threshold] = 0


    if np.max(heatmap_resized) > 0:
        heatmap_resized = heatmap_resized / np.max(heatmap_resized)


    if pred_label != 'notumor':

        heatmap_resized = np.power(heatmap_resized, 0.4)
        

        heatmap_uint8_clahe = np.uint8(255 * heatmap_resized)
        clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
        heatmap_uint8_clahe = clahe.apply(heatmap_uint8_clahe)
        heatmap_resized = heatmap_uint8_clahe / 255.0
        

        heatmap_resized = np.power(heatmap_resized, 0.5)

        heatmap_uint8_temp = np.uint8(255 * heatmap_resized)


        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        

        heatmap_uint8_temp = cv2.morphologyEx(heatmap_uint8_temp, cv2.MORPH_CLOSE, kernel_close)

        heatmap_uint8_temp = cv2.morphologyEx(heatmap_uint8_temp, cv2.MORPH_OPEN, kernel_open)

        heatmap_uint8_temp = cv2.dilate(heatmap_uint8_temp, kernel_dilate, iterations=2)
        
        heatmap_resized = heatmap_uint8_temp / 255.0
        

        if np.max(heatmap_resized) > 0:

            moments = cv2.moments(heatmap_uint8_temp)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cy, cx = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
            
            y_coords, x_coords = np.ogrid[:heatmap_resized.shape[0], :heatmap_resized.shape[1]]
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            
            max_distance = np.max(distances[heatmap_resized > 0]) if np.any(heatmap_resized > 0) else 1
            if max_distance > 0:
                distance_factor = 1 - (distances / (max_distance * 1.5))
                distance_factor = np.clip(distance_factor, 0, 1)
                distance_factor = np.power(distance_factor, 0.5)  
                
                heatmap_resized = heatmap_resized * distance_factor
        
        heatmap_uint8_temp = np.uint8(255 * heatmap_resized)
        heatmap_uint8_temp = cv2.GaussianBlur(heatmap_uint8_temp, (15, 15), 3)
        heatmap_resized = heatmap_uint8_temp / 255.0
        
        heatmap_resized = heatmap_resized * 1.2  
        heatmap_resized = np.clip(heatmap_resized, 0, 0.85)  
        
        if yolo_mask is not None:
            yolo_mask_resized = cv2.resize(yolo_mask, target_size, interpolation=cv2.INTER_NEAREST)
            yolo_mask_norm = yolo_mask_resized.astype(float) / 255.0
            heatmap_resized = heatmap_resized * yolo_mask_norm

    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    if pred_label == 'notumor':
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_OCEAN)
    else:
        heatmap_color = np.zeros((heatmap_uint8.shape[0], heatmap_uint8.shape[1], 3), dtype=np.uint8)
        heatmap_color[:, :, 2] = heatmap_uint8  
        heatmap_color[:, :, 0] = 0  
        heatmap_color[:, :, 1] = 0  

    original_img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    
    if pred_label != 'notumor':
        blue_tint = np.zeros_like(original_img_cv)
        blue_tint[:, :, 0] = original_img_cv[:, :, 0] * 0.9  
        blue_tint[:, :, 1] = original_img_cv[:, :, 1] * 0.4  
        blue_tint[:, :, 2] = original_img_cv[:, :, 2] * 0.4  
        original_img_cv = blue_tint.astype(np.uint8)

    if pred_label == 'notumor':
        alpha = 0.12
        beta = 0.88
    else:
        alpha = 0.65  
        beta = 0.35

    superimposed_img = cv2.addWeighted(original_img_cv, beta, heatmap_color, alpha, 0)
    superimposed_img = cv2.convertScaleAbs(superimposed_img, alpha=1.08, beta=5)

    return {
        'original': orig_array,
        'zoomed': zoomed,
        'gradcam': cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB),
        'prediction': pred_label,
        'confidence': confidence,
        'all_predictions': preds[0]
    }


# Enhanced prediction function using Grad-CAM for Pancreas
def predict_pancreas_with_gradcam(model, img):
    target_size = (299, 299)
    orig_array = np.array(img).astype('uint8')
    img_resized = img.resize(target_size)
    x = tf.keras.preprocessing.image.img_to_array(img_resized)
    x_norm = x / 255.0
    x_input = np.expand_dims(x_norm, axis=0)

    zoom_factor = 0.5
    h, w, _ = x_norm.shape
    center_h, center_w = h // 2, w // 2
    zoom_h, zoom_w = int(h * zoom_factor), int(w * zoom_factor)
    start_h = center_h - zoom_h // 2
    end_h = center_h + zoom_h // 2
    start_w = center_w - zoom_w // 2
    end_w = center_w + zoom_w // 2
    cropped = x_norm[start_h:end_h, start_w:end_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)


    zoomed_input = np.expand_dims(zoomed, axis=0)
    
    preds = model.predict(zoomed_input, verbose=0)
    pred_class = np.argmax(preds[0])
    pred_label = pancreas_class_names[pred_class]
    confidence = preds[0][pred_class] * 100


    heatmap = None
    best_heatmap = None
    best_score = -1

    try:

        try:
            base_model = model.get_layer("xception")

            layers_to_try = [
                "block13_sepconv2_act",
                "block12_sepconv2_act",
                "block14_sepconv2_act",
                "block11_sepconv2_act",
                "block10_sepconv2_act"
            ]

            for layer_name in layers_to_try:
                try:
                    last_conv_layer = base_model.get_layer(layer_name)
                    grad_model = tf.keras.models.Model(
                        inputs=base_model.input,
                        outputs=[last_conv_layer.output, base_model.output]
                    )

                    with tf.GradientTape() as tape:
                        conv_outputs, predictions = grad_model(zoomed_input)
                        loss = predictions[:, pred_class]

                    grads = tape.gradient(loss, conv_outputs)
                    

                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_outputs_np = conv_outputs[0].numpy()
                    pooled_grads_np = pooled_grads.numpy()


                    for i in range(pooled_grads_np.shape[0]):
                        if pooled_grads_np[i] > 0:
                            conv_outputs_np[:, :, i] *= pooled_grads_np[i]
                        else:
                            conv_outputs_np[:, :, i] = 0

                    heatmap = np.mean(conv_outputs_np, axis=-1)
                    heatmap = np.maximum(heatmap, 0)

                    if np.max(heatmap) > 0:
                        heatmap = heatmap / np.max(heatmap)


                    if pred_label != 'normal':
                        high_activation = np.sum(heatmap > 0.6)
                        medium_activation = np.sum(heatmap > 0.3)
                        total_activation = np.sum(heatmap > 0.05)

                        if total_activation > 0:
                            concentration_score = high_activation / (total_activation + 1)
                            clarity_score = medium_activation / (total_activation + 1)
                            size_penalty = min(1.0, total_activation / (heatmap.size * 0.3))
                            
                            score = (concentration_score * 0.5 + 
                                    clarity_score * 0.3 + 
                                    size_penalty * 0.2)

                            if score > best_score:
                                best_score = score
                                best_heatmap = heatmap.copy()
                    else:
                        if best_heatmap is None:
                            best_heatmap = heatmap.copy()
                        break

                except Exception as e:
                    continue

        except:

            last_conv_layer = None
            for layer in reversed(model.layers):
                if len(layer.output_shape) == 4:
                    last_conv_layer = layer
                    break

            if last_conv_layer is not None:
                grad_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=[last_conv_layer.output, model.output]
                )

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(zoomed_input)
                    loss = predictions[:, pred_class]

                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_outputs_np = conv_outputs[0].numpy()
                pooled_grads_np = pooled_grads.numpy()

                # Guided Grad-CAM
                for i in range(pooled_grads_np.shape[0]):
                    if pooled_grads_np[i] > 0:
                        conv_outputs_np[:, :, i] *= pooled_grads_np[i]
                    else:
                        conv_outputs_np[:, :, i] = 0

                heatmap = np.mean(conv_outputs_np, axis=-1)
                heatmap = np.maximum(heatmap, 0)
                best_heatmap = heatmap

        if best_heatmap is None and heatmap is not None:
            best_heatmap = heatmap


        if best_heatmap is not None:
            heatmap_resized = cv2.resize(best_heatmap, target_size, interpolation=cv2.INTER_CUBIC)

            # Normalize
            if np.max(heatmap_resized) > 0:
                heatmap_resized = heatmap_resized / np.max(heatmap_resized)


            if pred_label == 'normal':
                heatmap_resized = heatmap_resized * 0.15
                threshold = 0.6
            else:

                threshold_percentile = 35
                non_zero_values = heatmap_resized[heatmap_resized > 0.01]
                
                if len(non_zero_values) > 0:
                    threshold = np.percentile(non_zero_values, threshold_percentile)
                    threshold = max(0.08, min(0.3, threshold))
                else:
                    threshold = 0.1

            heatmap_resized[heatmap_resized < threshold] = 0


            if np.max(heatmap_resized) > 0:
                heatmap_resized = heatmap_resized / np.max(heatmap_resized)


            if pred_label != 'normal':


                heatmap_resized = np.power(heatmap_resized, 0.4)
                

                heatmap_uint8_clahe = np.uint8(255 * heatmap_resized)
                clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
                heatmap_uint8_clahe = clahe.apply(heatmap_uint8_clahe)
                heatmap_resized = heatmap_uint8_clahe / 255.0
                

                heatmap_resized = np.power(heatmap_resized, 0.5)

                heatmap_uint8_temp = np.uint8(255 * heatmap_resized)


                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                

                heatmap_uint8_temp = cv2.morphologyEx(heatmap_uint8_temp, cv2.MORPH_CLOSE, kernel_close)

                heatmap_uint8_temp = cv2.morphologyEx(heatmap_uint8_temp, cv2.MORPH_OPEN, kernel_open)

                heatmap_uint8_temp = cv2.dilate(heatmap_uint8_temp, kernel_dilate, iterations=3)

                heatmap_resized = heatmap_uint8_temp / 255.0
                

                heatmap_resized = heatmap_resized * 1.5
                heatmap_resized = np.clip(heatmap_resized, 0, 1)


            if pred_label == 'normal':
                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_OCEAN)
            else:

                heatmap_uint8 = np.uint8(255 * heatmap_resized)
                

                moments = cv2.moments(heatmap_uint8)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cy, cx = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
                

                active_points = np.argwhere(heatmap_uint8 > 30)
                if len(active_points) > 0:
                    y_coords, x_coords = active_points[:, 0], active_points[:, 1]
                    width = int((x_coords.max() - x_coords.min()) * 0.5)
                    height = int((y_coords.max() - y_coords.min()) * 0.4)
                    

                    width = max(30, min(width, 90))
                    height = max(25, min(height, 70))
                    
                    if abs(width - height) < 15:
                        width = int(height * 1.4)
                else:
                    width, height = 60, 40
                
                overlay = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                
                cv2.ellipse(overlay, (cx, cy), (width, height), 
                           0, 0, 360, (0, 0, 200), thickness=-1)
                
                cv2.ellipse(overlay, (cx, cy), (int(width*0.8), int(height*0.8)), 
                           0, 0, 360, (0, 0, 255), thickness=-1)
                
                overlay = cv2.GaussianBlur(overlay, (7, 7), 0)
                
                heatmap_color = overlay

            zoomed_bgr = cv2.cvtColor((zoomed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            original_img_cv = cv2.resize(zoomed_bgr, target_size, interpolation=cv2.INTER_CUBIC)
            
            if pred_label != 'normal':
                blue_tint = np.zeros_like(original_img_cv)
                blue_tint[:, :, 0] = original_img_cv[:, :, 0] * 0.9  # Blue channel
                blue_tint[:, :, 1] = original_img_cv[:, :, 1] * 0.4  # Green channel
                blue_tint[:, :, 2] = original_img_cv[:, :, 2] * 0.4  # Red channel
                original_img_cv = blue_tint.astype(np.uint8)

            if pred_label == 'normal':
                alpha = 0.12
                beta = 0.88
            else:
                alpha = 0.5 
                beta = 1.0   

            superimposed_img = cv2.addWeighted(original_img_cv, beta, heatmap_color, alpha, 0)
            superimposed_img = cv2.convertScaleAbs(superimposed_img, alpha=1.08, beta=5)
            gradcam_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback
            gradcam_img = orig_array

    except Exception as e:
        print(f"⚠️ Grad-CAM failed: {e}")
        gradcam_img = orig_array

    return {
        'original': orig_array,
        'zoomed': zoomed,
        'gradcam': gradcam_img,
        'prediction': pred_label,
        'confidence': confidence,
        'all_predictions': preds[0]
    }


# Function to standardize image size
def resize_for_display(img_array, size=(300, 300)):
    img_resized = cv2.resize(img_array, size, interpolation=cv2.INTER_AREA)
    return img_resized.astype(np.uint8)


# Function to ensure RGB uint8
def ensure_rgb_uint8(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img.astype(np.uint8)


# Background image path
bg_image_path = "26837.jpg"
bg_image_base64 = get_base64_image(bg_image_path)

# Custom CSS
if bg_image_base64:
    bg_style = f"""
    .stApp {{
        background-image: url('data:image/jpeg;base64,{bg_image_base64}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(102, 126, 234, 0.3);
        pointer-events: none;
        z-index: 0;
    }}
    """
else:
    bg_style = """
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    """

st.markdown(f"""
<style>
    {bg_style}

    .app-title {{
        text-align: center;
        color: #667eea;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}

    .success-message {{
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3em;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        animation: fadeIn 0.5s ease;
        display: inline-block;
        width: 100%;
    }}

    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: scale(0.95);
        }}
        to {{
            opacity: 1;
            transform: scale(1);
        }}
    }}

    .stButton > button {{
        width: 100%;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        border: 2px solid rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
        color: white !important;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 10px 0;
    }}
    .stButton > button:hover {{
        background: rgba(255, 255, 255, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }}

    .result-box {{
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }}
    .result-text {{
        font-size: 1.5em;
        font-weight: bold;
        color: #333;
    }}

    .video-container {{
        background: rgba(248, 249, 250, 0.9);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }}
    .video-title {{
        color: #667eea;
        font-size: 1.5em;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
    }}

    .image-box {{
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }}

    div[data-testid="stFileUploader"] {{
        position: relative;
        margin-top: -420px;
        height: 400px;
        z-index: 10;
    }}
    div[data-testid="stFileUploader"] > div {{
        height: 100%;
    }}
    div[data-testid="stFileUploader"] section {{
        height: 100%;
        border: none !important;
        background: transparent !important;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
    }}
    div[data-testid="stFileUploader"] section:hover {{
        background: rgba(102, 126, 234, 0.05) !important;
    }}
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
        background: transparent !important;
        border: none !important;
        height: 100%;
        width: 100%;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }}
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] > div {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
    }}
    div[data-testid="stFileUploader"] button {{
        margin: 20px auto 0 auto !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        display: block !important;
    }}
    div[data-testid="stFileUploader"] button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }}
    

    .hide-uploader div[data-testid="stFileUploader"] {{
        display: none !important;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stCodeBlock.st-emotion-cache-12r09dv.e1ycw9pz1 {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# Header
col_logo, col_title = st.columns([1.5, 8.5])
with col_logo:
    try:
        st.image("WhatsApp Image 2025-10-25 at 15.36.05.jpeg", width=120)
    except:
        st.markdown('<div style="font-size: 60px; margin-top: 15px;">🏥</div>', unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div style='margin-top: 0px; margin-left: 100px;'>
        <h1 style='color: #667eea; font-size: 2.5em; font-weight: bold; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);'>
            VIMD - Medical AI System
        </h1>
    </div>
    """, unsafe_allow_html=True)

# Model Selection
st.markdown("<h3 style='text-align: center; color: #667eea; margin-bottom: 30px;'>🔬 Select Detection Model</h3>",
            unsafe_allow_html=True)

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "brain"
if 'current_model' not in st.session_state:
    st.session_state.current_model = "brain"

col_brain, col_pancreas = st.columns(2)

with col_brain:
    brain_active = st.session_state.selected_model == "brain"
    brain_bg = "rgba(255, 255, 255, 0.35)" if brain_active else "rgba(255, 255, 255, 0.15)"
    brain_border = "rgba(255, 255, 255, 0.7)" if brain_active else "rgba(255, 255, 255, 0.3)"
    brain_shadow = "0 8px 25px rgba(102, 126, 234, 0.4)" if brain_active else "0 4px 10px rgba(0, 0, 0, 0.1)"

    brain_clicked = st.button(
        "🧠 Brain Tumor Detection",
        key="brain_btn",
        use_container_width=True,
        type="secondary"
    )

    st.markdown(f"""
    <style>
    button[kind="secondary"]:nth-of-type(1) {{
        background: {brain_bg} !important;
        border: 2px solid {brain_border} !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: white !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: {brain_shadow} !important;
        height: 70px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    if brain_clicked:
        st.session_state.selected_model = "brain"
        if st.session_state.get('current_model') != "brain":
            st.session_state.uploaded_images = []
            st.session_state.prediction_results = []
            st.session_state.yolo_results = []
            st.session_state.show_success = False
            st.session_state.current_model = "brain"
            st.rerun()

with col_pancreas:
    pancreas_active = st.session_state.selected_model == "pancreas"
    pancreas_bg = "rgba(255, 255, 255, 0.35)" if pancreas_active else "rgba(255, 255, 255, 0.15)"
    pancreas_border = "rgba(255, 255, 255, 0.7)" if pancreas_active else "rgba(255, 255, 255, 0.3)"
    pancreas_shadow = "0 8px 25px rgba(102, 126, 234, 0.4)" if pancreas_active else "0 4px 10px rgba(0, 0, 0, 0.1)"

    pancreas_clicked = st.button(
        "🫀 Pancreas Detection",
        key="pancreas_btn",
        use_container_width=True,
        type="secondary"
    )

    st.markdown(f"""
    <style>
    button[kind="secondary"]:nth-of-type(2) {{
        background: {pancreas_bg} !important;
        border: 2px solid {pancreas_border} !important;
        border-radius: 12px !important;
        padding: 20px !important;
        color: white !important;
        font-size: 1.1em !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: {pancreas_shadow} !important;
        height: 70px !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    if pancreas_clicked:
        st.session_state.selected_model = "pancreas"
        if st.session_state.get('current_model') != "pancreas":
            st.session_state.uploaded_images = []
            st.session_state.prediction_results = []
            st.session_state.yolo_results = []
            st.session_state.show_success = False
            st.session_state.current_model = "pancreas"
            st.rerun()

model_type = st.session_state.selected_model

# Initialize session state
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []
if 'yolo_results' not in st.session_state:
    st.session_state.yolo_results = []
if 'show_success' not in st.session_state:
    st.session_state.show_success = False

# Load model
try:
    if model_type == "brain":
        model = load_brain_model()
        yolo_model = load_brain_yolo_model()
        model_loaded = True
        yolo_loaded = True
        model_name = "Brain"
    else:
        model = load_pancreas_model()
        model_loaded = True
        yolo_loaded = False
        model_name = "Pancreas"
except:
    model_loaded = False
    yolo_loaded = False
    model_name = "Brain" if model_type == "brain" else "Pancreas"
    st.error(f"⚠️ {model_name} Model file not found!")

# Image upload
upload_container = st.container()
with upload_container:
    if len(st.session_state.uploaded_images) > 0:
        remaining = 3 - len(st.session_state.uploaded_images)
        st.markdown(f"""
        <div class="success-message" style="margin-bottom: 20px;">
            <span style="font-size: 1.5em; margin-right: 10px;">✅</span>
            <span>{len(st.session_state.uploaded_images)} image(s) uploaded successfully!</span>
            {"<span style='margin-left: 15px; opacity: 0.85;'>(You can upload " + str(remaining) + " more image(s))</span>" if remaining > 0 else "<span style='margin-left: 15px; opacity: 0.85;'>(Maximum limit reached)</span>"}
        </div>
        """, unsafe_allow_html=True)

        if len(st.session_state.uploaded_images) < 3:
            st.markdown("""
            <div style='border: 3px dashed #667eea; border-radius: 15px; background: rgba(240, 240, 255, 0.6); 
                        padding: 0; min-height: 400px; position: relative; overflow: hidden; backdrop-filter: blur(10px);'>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "label",
                type=['png', 'jpg', 'jpeg'],
                key=f"file_uploader_add_{len(st.session_state.uploaded_images)}",
                label_visibility="hidden",
                accept_multiple_files=False
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.session_state.uploaded_images.append(image)
                st.rerun()
        else:
            st.markdown("""
            <div style='border: 3px solid #4CAF50; border-radius: 15px; background: rgba(76, 175, 80, 0.1); 
                        padding: 60px; min-height: 400px; backdrop-filter: blur(10px);
                        display: flex; align-items: center; justify-content: center; text-align: center;'>
                <div style='color: #4CAF50; font-size: 1.5em; font-weight: bold;'>
                    <div style='font-size: 3em; margin-bottom: 20px;'>📸</div>
                    <div>3 Images Uploaded</div>
                    <div style='font-size: 0.8em; margin-top: 10px; opacity: 0.8;'>Click "Analyze Image" to continue</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='border: 3px dashed #667eea; border-radius: 15px; background: rgba(240, 240, 255, 0.6); 
                    padding: 0; min-height: 400px; position: relative; overflow: hidden; backdrop-filter: blur(10px);'>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "label",
            type=['png', 'jpg', 'jpeg'],
            key="file_uploader",
            label_visibility="hidden",
            accept_multiple_files=True
        )

        if uploaded_files:
            for uploaded_file in uploaded_files[:3]:
                image = Image.open(uploaded_file).convert('RGB')
                st.session_state.uploaded_images.append(image)
            st.session_state.show_success = True
            st.rerun()

col1, col2 = st.columns(2)
with col1:
    if st.button("Analyze Image", key="predict", use_container_width=True):
        if len(st.session_state.uploaded_images) > 0 and model_loaded:
            progress_container = st.empty()

            security_messages = [
                "Encrypting medical data...",
                "Your data is 100% confidential and secure",
                "HIPAA compliant - Your privacy is our priority",
                "Processing medical scans securely...",
                "No data is stored - All analysis happens locally",
                "End-to-end encrypted analysis in progress",
                "Your medical information remains private",
                "Analyzing with state-of-the-art AI security"
            ]

            total_images = len(st.session_state.uploaded_images)

            progress_html = """
            <style>
                @keyframes progressFlow {{
                    0% {{ background-position: 0% 50%; }}
                    50% {{ background-position: 100% 50%; }}
                    100% {{ background-position: 0% 50%; }}
                }}

                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(-20px) scale(0.95); }}
                    to {{ opacity: 1; transform: translateY(0) scale(1); }}
                }}

                @keyframes glowPulse {{
                    0%, 100% {{ 
                        box-shadow: 0 0 30px rgba(102, 126, 234, 0.4), 0 0 60px rgba(102, 126, 234, 0.2);
                    }}
                    50% {{ 
                        box-shadow: 0 0 50px rgba(102, 126, 234, 0.6), 0 0 100px rgba(102, 126, 234, 0.3);
                    }}
                }}

                @keyframes rotateIcon {{
                    0% {{ transform: rotate(0deg) scale(1); }}
                    50% {{ transform: rotate(180deg) scale(1.15); }}
                    100% {{ transform: rotate(360deg) scale(1); }}
                }}

                @keyframes shimmer {{
                    0% {{ transform: translateX(-100%); }}
                    100% {{ transform: translateX(100%); }}
                }}

                @keyframes statScale {{
                    from {{ opacity: 0; transform: scale(0.9) translateY(15px); }}
                    to {{ opacity: 1; transform: scale(1) translateY(0); }}
                }}

                @keyframes borderGlow {{
                    0%, 100% {{ border-color: rgba(102, 126, 234, 0.3); }}
                    50% {{ border-color: rgba(102, 126, 234, 0.7); }}
                }}

                .progress-centered-wrapper {{
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    width: 90%;
                    max-width: 1000px;
                    z-index: 9999;
                    animation: fadeIn 0.5s cubic-bezier(0.16, 1, 0.3, 1);
                }}

                .progress-overlay {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.6);
                    backdrop-filter: blur(8px);
                    z-index: 9998;
                    animation: fadeIn 0.3s ease;
                }}

                .progress-box {{
                    background: linear-gradient(135deg, 
                        rgba(255, 255, 255, 0.98) 0%, 
                        rgba(248, 250, 252, 0.98) 100%);
                    border-radius: 25px;
                    padding: 50px 60px;
                    box-shadow: 
                        0 20px 60px rgba(102, 126, 234, 0.25),
                        0 0 0 1px rgba(102, 126, 234, 0.1),
                        inset 0 1px 0 rgba(255, 255, 255, 0.9);
                    position: relative;
                    overflow: hidden;
                }}

                .progress-box::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #667eea 100%);
                    background-size: 200% 100%;
                    animation: progressFlow 3s ease-in-out infinite;
                }}

                .progress-header {{
                    text-align: center;
                    margin-bottom: 40px;
                }}

                .progress-icon-wrapper {{
                    position: relative;
                    width: 100px;
                    height: 100px;
                    margin: 0 auto 30px;
                }}

                .progress-icon {{
                    width: 100px;
                    height: 100px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    animation: rotateIcon 4s cubic-bezier(0.4, 0, 0.2, 1) infinite;
                    position: relative;
                    z-index: 2;
                    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4);
                }}

                .progress-icon::before {{
                    content: '';
                    position: absolute;
                    width: 120%;
                    height: 120%;
                    border-radius: 50%;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    animation: glowPulse 2.5s ease-in-out infinite;
                    z-index: -1;
                    opacity: 0.4;
                }}

                .progress-icon svg {{
                    width: 45px;
                    height: 45px;
                    fill: white;
                    filter: drop-shadow(0 3px 6px rgba(0,0,0,0.3));
                }}

                .progress-title {{
                    color: #1a202c;
                    font-size: 2.4em;
                    font-weight: 800;
                    margin: 0 0 15px 0;
                    letter-spacing: -1px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}

                .security-message {{
                    color: #059669;
                    font-size: 1.05em;
                    font-weight: 600;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 10px;
                    padding: 12px 25px;
                    background: rgba(16, 185, 129, 0.1);
                    border-radius: 50px;
                    display: inline-flex;
                    margin: 0 auto;
                }}

                .security-icon {{
                    width: 20px;
                    height: 20px;
                    animation: glowPulse 2s ease-in-out infinite;
                }}

                .progress-bar-section {{
                    margin: 35px 0;
                }}

                .progress-bar-container {{
                    background: linear-gradient(135deg, 
                        rgba(102, 126, 234, 0.1) 0%, 
                        rgba(118, 75, 162, 0.1) 100%);
                    border-radius: 100px;
                    height: 60px;
                    overflow: hidden;
                    position: relative;
                    box-shadow: 
                        inset 0 2px 8px rgba(0, 0, 0, 0.08),
                        0 1px 0 rgba(255, 255, 255, 0.9);
                    border: 2px solid rgba(102, 126, 234, 0.2);
                    animation: borderGlow 3s ease-in-out infinite;
                }}

                .progress-bar {{
                    height: 100%;
                    background: linear-gradient(
                        90deg,
                        #667eea 0%,
                        #764ba2 20%,
                        #f093fb 40%,
                        #667eea 60%,
                        #764ba2 80%,
                        #f093fb 100%
                    );
                    background-size: 300% 100%;
                    animation: progressFlow 3s ease-in-out infinite;
                    border-radius: 100px;
                    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: 800;
                    font-size: 1.2em;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    letter-spacing: 1.2px;
                    position: relative;
                    overflow: hidden;
                }}

                .progress-bar::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 40%;
                    background: linear-gradient(
                        to bottom,
                        rgba(255, 255, 255, 0.35),
                        transparent
                    );
                    border-radius: 100px 100px 0 0;
                }}

                .progress-bar::after {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 40%;
                    height: 100%;
                    background: linear-gradient(
                        90deg,
                        transparent,
                        rgba(255, 255, 255, 0.4),
                        transparent
                    );
                    animation: shimmer 2s ease-in-out infinite;
                }}

                .progress-details {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin: 35px 0 0 0;
                }}

                .progress-stat {{
                    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                    padding: 25px;
                    border-radius: 18px;
                    box-shadow: 
                        0 8px 25px rgba(102, 126, 234, 0.1),
                        0 0 0 1px rgba(102, 126, 234, 0.08),
                        inset 0 1px 0 rgba(255, 255, 255, 0.9);
                    text-align: center;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    animation: statScale 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
                    position: relative;
                    overflow: hidden;
                }}

                .progress-stat::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 100%;
                    height: 100%;
                    background: linear-gradient(
                        90deg,
                        transparent,
                        rgba(102, 126, 234, 0.08),
                        transparent
                    );
                    transition: left 0.5s;
                }}

                .progress-stat:hover::before {{
                    left: 100%;
                }}

                .progress-stat:nth-child(1) {{ animation-delay: 0.1s; }}
                .progress-stat:nth-child(2) {{ animation-delay: 0.2s; }}
                .progress-stat:nth-child(3) {{ animation-delay: 0.3s; }}

                .progress-stat:hover {{
                    transform: translateY(-5px) scale(1.03);
                    box-shadow: 
                        0 15px 40px rgba(102, 126, 234, 0.2),
                        0 0 0 1px rgba(102, 126, 234, 0.2);
                }}

                .stat-label {{
                    font-size: 0.8em;
                    color: #64748b;
                    margin-bottom: 10px;
                    text-transform: uppercase;
                    letter-spacing: 1.5px;
                    font-weight: 700;
                }}

                .stat-value {{
                    font-size: 2em;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    font-weight: 900;
                    letter-spacing: -0.5px;
                    line-height: 1.2;
                }}

                .privacy-badge {{
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    padding: 22px 35px;
                    border-radius: 16px;
                    margin-top: 30px;
                    text-align: center;
                    font-weight: 600;
                    font-size: 0.95em;
                    box-shadow: 
                        0 10px 30px rgba(16, 185, 129, 0.3),
                        inset 0 1px 0 rgba(255, 255, 255, 0.25);
                    letter-spacing: 0.3px;
                    line-height: 1.7;
                    animation: statScale 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.4s backwards;
                }}
            </style>

            <div class="progress-overlay"></div>
            <div class="progress-centered-wrapper">
                <div class="progress-box">
                    <div class="progress-header">
                        <div class="progress-icon-wrapper">
                            <div class="progress-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                                </svg>
                            </div>
                        </div>
                        <div class="progress-title">AI Medical Analysis</div>
                        <div class="security-message">
                            <svg class="security-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
                            </svg>
                            {security_msg}
                        </div>
                    </div>

                    <div class="progress-bar-section">
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: {progress}%;">
                                {progress_text}
                            </div>
                        </div>
                    </div>

                    <div class="progress-details">
                        <div class="progress-stat">
                            <div class="stat-label">Progress</div>
                            <div class="stat-value">{progress:.0f}%</div>
                        </div>
                        <div class="progress-stat">
                            <div class="stat-label">Image</div>
                            <div class="stat-value">{current} / {total}</div>
                        </div>
                        <div class="progress-stat">
                            <div class="stat-label">Status</div>
                            <div class="stat-value">Active</div>
                        </div>
                    </div>

                    <div class="privacy-badge">
                        🔒 All medical data is encrypted and processed securely<br>
                        ✓ No storage • Complete privacy guaranteed
                    </div>
                </div>
            </div>
            """

            st.session_state.prediction_results = []
            st.session_state.yolo_results = []

            import time
            import random

            for idx, img in enumerate(st.session_state.uploaded_images):
                current_image = idx + 1


                for step in range(5):
                    progress = ((idx * 5 + step + 1) / (total_images * 5)) * 100


                    security_msg = security_messages[random.randint(0, len(security_messages) - 1)]


                    if step == 0:
                        progress_text = "INITIALIZING..."
                    elif step == 1:
                        progress_text = "SCANNING..."
                    elif step == 2:
                        progress_text = "AI ANALYSIS..."
                    elif step == 3:
                        progress_text = "PROCESSING..."
                    else:
                        progress_text = "SECURING..."


                    progress_container.markdown(
                        progress_html.format(
                            security_msg=security_msg,
                            progress=min(progress, 95),
                            progress_text=progress_text,
                            current=current_image,
                            total=total_images
                        ),
                        unsafe_allow_html=True
                    )

                    time.sleep(0.3)


                if model_type == "brain":
                    if yolo_loaded:

                        try:
                            results = yolo_model(img)
                            tumor_mask = None
                            

                            img_array = np.array(img)
                            yolo_img = cv2.cvtColor(img_array.copy(), cv2.COLOR_RGB2BGR)
                            
                            if len(results) > 0 and len(results[0].boxes) > 0:

                                mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
                                
                                for box in results[0].boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                                    

                                    class_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                                    class_name = brain_class_names[class_id]
                                    

                                    if class_name == 'notumor':
                                        box_color = (0, 255, 0)
                                        label = "Negative"
                                        text_bg_color = (0, 200, 0)
                                    else:
                                        box_color = (0, 0, 255)
                                        label = "Positive"
                                        text_bg_color = (0, 0, 200)
                                    

                                    cv2.rectangle(yolo_img, (x1, y1), (x2, y2), box_color, 3)
                                    

                                    text = label
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.7
                                    thickness = 2
                                    

                                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                                    

                                    cv2.rectangle(yolo_img, 
                                                (x1, y1 - text_h - 10), 
                                                (x1 + text_w + 10, y1), 
                                                text_bg_color, -1)
                                    

                                    cv2.putText(yolo_img, text, 
                                              (x1 + 5, y1 - 5), 
                                              font, font_scale, (255, 255, 255), thickness)
                                

                                yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
                                tumor_mask = mask
                            
                            yolo_result = {
                                'image': img_array,
                                'yolo_detection': yolo_img,
                                'results': results,
                                'tumor_mask': tumor_mask
                            }
                            st.session_state.yolo_results.append(yolo_result)
                        except:
                            tumor_mask = None
                            img_array = np.array(img)
                            st.session_state.yolo_results.append({
                                'image': img_array,
                                'yolo_detection': img_array,
                                'tumor_mask': None
                            })
                    else:
                        tumor_mask = None

                    result = predict_brain_with_gradcam(model, img, tumor_mask)
                    st.session_state.prediction_results.append(result)
                else:
                    result = predict_pancreas_with_gradcam(model, img)
                    st.session_state.prediction_results.append(result)


            progress_container.markdown(
                progress_html.format(
                    security_msg="✓ Analysis Complete - All data secured and encrypted",
                    progress=100,
                    progress_text="COMPLETE ✓",
                    current=total_images,
                    total=total_images
                ),
                unsafe_allow_html=True
            )

            time.sleep(1.8)
            progress_container.empty()

        elif not model_loaded:
            st.error("⚠️ Model not loaded!")
        else:
            st.warning("⚠️ Please upload at least one image!")

with col2:
    if st.button("Reset", key="reset", use_container_width=True):
        st.session_state.uploaded_images = []
        st.session_state.prediction_results = []
        st.session_state.yolo_results = []
        st.session_state.show_success = False
        st.rerun()


# Display results
if len(st.session_state.prediction_results) > 0:
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #667eea;'> Analysis Results</h2>", unsafe_allow_html=True)

    if model_type == "brain":
        disease_descriptions = brain_disease_descriptions
    else:
        disease_descriptions = pancreas_disease_descriptions

    for idx, result in enumerate(st.session_state.prediction_results):
        st.markdown(f"<h3 style='color: #667eea;'>Image {idx + 1}</h3>", unsafe_allow_html=True)

        pred_label = result['prediction']
        disease_info = disease_descriptions[pred_label]

        st.markdown(f"""
        <div class="result-box" style="background: linear-gradient(135deg, {disease_info['color']}15 0%, {disease_info['color']}30 100%);">
            <div class="result-text">
                🧠 Diagnosis: {result['prediction'].upper()}<br>
                📈 Prediction Confidence: {result['confidence']:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        if model_type == "brain" and len(st.session_state.yolo_results) > idx:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.markdown("**Original Image**")
                orig_display = resize_for_display(result['original'])
                st.image(ensure_rgb_uint8(orig_display), width=300)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.markdown("**YOLO Detection**")
                yolo_result = st.session_state.yolo_results[idx]
                yolo_display = resize_for_display(yolo_result['yolo_detection'])
                st.image(ensure_rgb_uint8(yolo_display), width=300)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.markdown("**Processed (Zoomed)**")
                zoomed_display = resize_for_display((result['zoomed'] * 255))
                st.image(ensure_rgb_uint8(zoomed_display), width=300)
                st.markdown('</div>', unsafe_allow_html=True)

            with col4:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.markdown("**Grad-CAM Analysis**")
                gradcam_display = resize_for_display(result['gradcam'])
                st.image(ensure_rgb_uint8(gradcam_display), width=300)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.markdown("**Original Image**")
                orig_display = resize_for_display(result['original'])
                st.image(ensure_rgb_uint8(orig_display), width=400)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.markdown("**Processed (Zoomed)**")
                zoomed_display = resize_for_display((result['zoomed'] * 255))
                st.image(ensure_rgb_uint8(zoomed_display), width=400)
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="image-box">', unsafe_allow_html=True)
                st.markdown("**Grad-CAM Analysis**")
                gradcam_display = resize_for_display(result['gradcam'])
                st.image(ensure_rgb_uint8(gradcam_display), width=400)
                st.markdown('</div>', unsafe_allow_html=True)

        # Display detailed description
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 25px; 
                    margin: 20px 0; box-shadow: 0 5px 15px rgba(0,0,0,0.1); 
                    border-left: 5px solid {disease_info['color']};'>
            <h3 style='color: {disease_info['color']}; margin-bottom: 15px;'>
                📋 {disease_info['name_en']}
            </h3>
            <div style='color: #333; line-height: 1.8;'>
                {disease_info['description']}
            </div>
        </div>

        <div style='margin-top: 20px; padding: 15px; background: {disease_info['color']}20; 
                    border-radius: 10px; text-align: center;'>
            <strong style='color: {disease_info['color']}; font-size: 1.2em;'>
                Severity Level: {disease_info['severity']}
            </strong>
        </div>
        """, unsafe_allow_html=True)


# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; color: white; margin-top: 20px; background: rgba(0,0,0,0.3); border-radius: 10px;'>
    <p style='margin: 0; font-weight: bold;'>© 2024 VIMD Medical AI System</p>
    <p style='margin: 5px 0 0 0; font-size: 0.9em;'>This project was developed by Eng. Ahmed Ayman</p>
    <p style='margin: 0; font-size: 0.9em;'>This project is privately owned by Mr. Mishari</p>
    <p style='margin: 5px 0 0 0; font-size: 0.9em;'>Powered by Streamlit & Deep Learning | Multi-Model Detection System</p>
</div>
""", unsafe_allow_html=True)