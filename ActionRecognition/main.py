import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Load your action recognition model
# Replace this with your actual model loading code
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('new.h5')  # Path to your trained model
    return model

# Class labels for UC50 dataset
action_labels = ['WalkingWithDog', 'TaiChi', 'Swing', 'HorseRace', 'Biking']
# Function to preprocess video frame
def preprocess_frame(frame, num_frames=20):
    # Resize frame to expected input shape of your model
    frame = cv2.resize(frame, (64, 64))
    # Convert frame to numpy array
    frame_array = np.array(frame)
    # Normalize pixel values
    frame_array = frame_array / 255.0
    return frame_array

# Function to make prediction on video
def predict_action_on_video(video_path, model, num_frames=20):
    video = cv2.VideoCapture(video_path)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
    
    frame_sequence = []  # Initialize an empty list to hold frames
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Preprocess frame
        preprocessed_frame = preprocess_frame(frame)
        frame_sequence.append(preprocessed_frame)
        
        # Keep only the last num_frames frames in the sequence
        if len(frame_sequence) > num_frames:
            frame_sequence = frame_sequence[-num_frames:]
        
        if len(frame_sequence) == num_frames:
            # Perform prediction
            predictions = model.predict(np.array([frame_sequence]))
            # Get the index of the class with the highest probability
            predicted_class_index = np.argmax(predictions)
            # Get the corresponding action label
            predicted_action = action_labels[predicted_class_index]
            # Get the confidence score
            confidence = predictions[0][predicted_class_index]
            
            # Draw predicted action on frame
            cv2.putText(frame, f"Action: {predicted_action}, Confidence: {confidence:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Write the frame with prediction to output video
            out.write(frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    st.title('Action Recognition using UCF50 ')
    st.sidebar.title('Upload Video')

    uploaded_video = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi"])

    if uploaded_video is not None:
        # Display the uploaded video
        st.video(uploaded_video)
        st.write("")

        # Load the model
        model = load_model()

        # Save the uploaded video to a temporary location
        with open("temp_video.avi", "wb") as f:
            f.write(uploaded_video.read())

        # Make prediction on video
        predict_action_on_video("temp_video.avi", model)

        # Display the processed video with action predictions
        st.video("output.avi")

if __name__ == '__main__':
    main()

