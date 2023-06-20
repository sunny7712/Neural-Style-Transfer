import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import streamlit as st

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(bytes_data):
    # img = tf.io.read_file(img_path)
    img = tf.image.decode_image(bytes_data, channels = 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

st.title("I'm something of an Artist Myself!")
st.write("")
st.write("")
col1, col2 = st.columns(2)
content_uploaded_file = st.file_uploader("Choose a content image", type = ['png', 'jpg', 'jpeg'])
if content_uploaded_file is not None:
    bytes_data = content_uploaded_file.getvalue()
    content_img_tensor = load_image(bytes_data)
    # st.write('{}'.format(type(img_tensor)))
    with col1:
        content_img_array = np.squeeze(np.array(content_img_tensor))
        # st.write('{}'.format(content_img_array.shape))
        st.image(content_img_array)
        st.write('Uploaded content image..')
st.write("")
st.write("")
style_uploaded_file = st.file_uploader("Choose a style image", type = ['png', 'jpg', 'jpeg'])
if style_uploaded_file is not None:
    bytes_data = style_uploaded_file.getvalue()
    style_img_tensor = load_image(bytes_data)
    # st.write('{}'.format(type(img_tensor)))
    with col2:
        style_img_array = np.squeeze(np.array(style_img_tensor))
        # st.write('{}'.format(content_img_array.shape))
        st.image(style_img_array)
        st.write('Uploaded style image..')
st.write("")
st.write("")



if content_uploaded_file is not None and style_uploaded_file is not None:
    stylized_image = model(tf.constant(content_img_tensor), tf.constant(style_img_tensor))[0]
    st.header("Stylized image:")
    st.image(np.squeeze(np.array(stylized_image)))
