import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    # load the images db and the facial keypoints df
    face_images_db = np.load('archive/face_images.npz')['face_images']
    facial_keypoints_df = pd.read_csv('archive/facial_keypoints.csv')

    # remove all the images that don't have complete data attached to them
    face_images_db, facial_keypoints_df = remove_incomplete_keypoints(face_images_db, facial_keypoints_df)

    if face_images_db.shape[2] > 1200:
        face_images_db = face_images_db[:,:, :1200]  # Keep only the first 1200 images

    np.savez('archive/face_images_updated.npz', face_images=face_images_db)

    get_number_of_faces(face_images_db)

    bb_df = generate_bb(facial_keypoints_df)
    print(bb_df)
    print_image(face_images_db, bb_df, 300)

    bb_df.to_csv("bb_csv.csv", index=False)

def print_image(face_images_db, bb_df, image_number):
    # Load the image (assuming face_images_db is already loaded)
    image = face_images_db[:,:,image_number]

    # Create a figure and axis to plot
    fig, ax = plt.subplots(figsize=(6,6))

    # Display the image
    ax.imshow(image, cmap='gray')

    # Create a rectangle for the left eye (position and size)
    left_eye_x = bb_df.loc[image_number, 'left_eye_x']
    left_eye_y = bb_df.loc[image_number, 'left_eye_y']
    left_eye_width = bb_df.loc[image_number, 'left_eye_width']
    left_eye_height = bb_df.loc[image_number, 'left_eye_height']
    left_eye_rect = patches.Rectangle(
        (left_eye_x - left_eye_width / 2, left_eye_y - left_eye_height / 2), 
        left_eye_width, 
        left_eye_height, 
        linewidth=2, 
        edgecolor='r', 
        facecolor='none'
    )
    ax.add_patch(left_eye_rect)

    # right eye
    right_eye_x = bb_df.loc[image_number, 'right_eye_x']
    right_eye_y = bb_df.loc[image_number, 'right_eye_y']
    right_eye_width = bb_df.loc[image_number, 'right_eye_width']
    right_eye_height = bb_df.loc[image_number, 'right_eye_height']
    right_eye_rect = patches.Rectangle(
        (right_eye_x - right_eye_width / 2, right_eye_y - right_eye_height / 2), 
        right_eye_width, 
        right_eye_height, 
        linewidth=2, 
        edgecolor='b', 
        facecolor='none'
    )
    ax.add_patch(right_eye_rect)

    # nose
    nose_x = bb_df.loc[image_number, 'nose_x']
    nose_y = bb_df.loc[image_number, 'nose_y']
    nose_width = bb_df.loc[image_number, 'nose_width']
    nose_height = bb_df.loc[image_number, 'nose_height']
    nose_rect = patches.Rectangle(
        (nose_x - nose_width / 2, nose_y - nose_height / 2), 
        nose_width, 
        nose_height, 
        linewidth=2, 
        edgecolor='g', 
        facecolor='none'
    )
    ax.add_patch(nose_rect)

    # mouth
    mouth_x = bb_df.loc[image_number, 'mouth_x']
    mouth_y = bb_df.loc[image_number, 'mouth_y']
    mouth_width = bb_df.loc[image_number, 'mouth_width']
    mouth_height = bb_df.loc[image_number, 'mouth_height']
    mouth_rect = patches.Rectangle(
        (mouth_x - mouth_width / 2, mouth_y - mouth_height / 2), 
        mouth_width, 
        mouth_height, 
        linewidth=2, 
        edgecolor='y', 
        facecolor='none'
    )
    ax.add_patch(mouth_rect)

    # Example for plotting the keypoints (eyes, nose, and mouth) as red dots
    ax.scatter([left_eye_x, right_eye_x, nose_x, mouth_x], 
            [left_eye_y, right_eye_y, nose_y, mouth_y], 
            color='red', s=50)

    # Remove axes for better display
    ax.axis('off')

    # Show the image with bounding boxes
    plt.show()
    
# print the number of faces in the dataset and the dimensions of the images
def get_number_of_faces(face_images_db):
    (imHeight, imWidth, numImages) = face_images_db.shape

    print('number of remaining images = %d' %(numImages))
    print('image dimentions = (%d,%d)' %(imHeight,imWidth))

# remove incomplete data from the dataset
def remove_incomplete_keypoints(face_images_db, facial_keypoints_df):
    numMissingKeypoints = facial_keypoints_df.isnull().sum(axis=1)
    allKeypointsPresentInds = np.nonzero(numMissingKeypoints == 0)[0]

    return face_images_db[:,:,allKeypointsPresentInds], facial_keypoints_df.iloc[allKeypointsPresentInds,:].reset_index(drop=True)

def generate_bb(facial_keypoints_df):
    data = {
    'left_eye_x': [],
    'left_eye_y': [],
    'left_eye_height': [],
    'left_eye_width': [],
    'right_eye_x': [],
    'right_eye_y': [],
    'right_eye_height': [],
    'right_eye_width': [],
    'nose_x': [],
    'nose_y': [],
    'nose_height': [],
    'nose_width': [],
    'mouth_x': [],
    'mouth_y': [],
    'mouth_height': [],
    'mouth_width': [],
    }

    bb_df = pd.DataFrame(data)

    for index, row in facial_keypoints_df.iterrows():
        next_row = calculate_bb(facial_keypoints_df, index)
        bb_df = pd.concat([bb_df, next_row], ignore_index=True)  # Append to bb_df

    return bb_df

def calculate_bb(facial_keypoints_df, image_number):
    data = {
    'left_eye_x': [],
    'left_eye_y': [],
    'left_eye_height': [],
    'left_eye_width': [],
    'right_eye_x': [],
    'right_eye_y': [],
    'right_eye_height': [],
    'right_eye_width': [],
    'nose_x': [],
    'nose_y': [],
    'nose_height': [],
    'nose_width': [],
    'mouth_x': [],
    'mouth_y': [],
    'mouth_height': [],
    'mouth_width': [],
    }

    # calculate left eye x
    left_eye_keypoints_x = ['left_eye_center_x', 'left_eye_inner_corner_x', 'left_eye_outer_corner_x']
    left_eye_mean_x = facial_keypoints_df.loc[image_number, left_eye_keypoints_x].mean()
    data['left_eye_x'].append(left_eye_mean_x)

    # calculate left eye width
    left_outer_eyebrow_x = -(left_eye_mean_x - facial_keypoints_df.loc[image_number, 'left_eyebrow_outer_end_x'])
    left_inner_eyebrow_x = left_eye_mean_x - facial_keypoints_df.loc[image_number, 'left_eyebrow_inner_end_x']
    data['left_eye_width'].append(1.1 * (left_inner_eyebrow_x + left_outer_eyebrow_x))

    # calculate left eye y
    left_eye_keypoints_y = ['left_eye_center_y', 'left_eye_inner_corner_y', 'left_eye_outer_corner_y']
    left_eye_mean_y = facial_keypoints_df.loc[image_number, left_eye_keypoints_y].mean()
    data['left_eye_y'].append(left_eye_mean_y)

    # calculate left eye height
    left_outer_eyebrow_y = left_eye_mean_y - facial_keypoints_df.loc[image_number, 'left_eyebrow_outer_end_y']
    left_inner_eyebrow_y = left_eye_mean_y - facial_keypoints_df.loc[image_number, 'left_eyebrow_inner_end_y']
    data['left_eye_height'].append(1.1 * (left_inner_eyebrow_y + left_outer_eyebrow_y))

    # calculate right eye x
    right_eye_keypoints_x = ['right_eye_center_x', 'right_eye_inner_corner_x', 'right_eye_outer_corner_x']
    right_eye_mean_x = facial_keypoints_df.loc[image_number, right_eye_keypoints_x].mean()
    data['right_eye_x'].append(right_eye_mean_x)

    # calculate right eye width
    right_outer_eyebrow_x = -(right_eye_mean_x - facial_keypoints_df.loc[image_number, 'right_eyebrow_outer_end_x'])
    right_inner_eyebrow_x = right_eye_mean_x - facial_keypoints_df.loc[image_number, 'right_eyebrow_inner_end_x']
    data['right_eye_width'].append(1.1 * (right_inner_eyebrow_x + right_outer_eyebrow_x))

    # calculate right eye y
    right_eye_keypoints_y = ['right_eye_center_y', 'right_eye_inner_corner_y', 'right_eye_outer_corner_y']
    right_eye_mean_y = facial_keypoints_df.loc[image_number, right_eye_keypoints_y].mean()
    data['right_eye_y'].append(right_eye_mean_y)

    # calculate right eye height
    right_outer_eyebrow_y = right_eye_mean_y - facial_keypoints_df.loc[image_number, 'right_eyebrow_outer_end_y']
    right_inner_eyebrow_y = right_eye_mean_y - facial_keypoints_df.loc[image_number, 'right_eyebrow_inner_end_y']
    data['right_eye_height'].append(1.1 * (right_inner_eyebrow_y + right_outer_eyebrow_y))

    # calculate nose x
    nose_keypoints_x = ['left_eye_inner_corner_x','right_eye_inner_corner_x','mouth_right_corner_x','mouth_left_corner_x','nose_tip_x']
    nose_mean_x = facial_keypoints_df.loc[image_number, nose_keypoints_x].mean()
    data['nose_x'].append(nose_mean_x)

    # calculate nose y
    nose_keypoints_y = ['left_eye_inner_corner_y','right_eye_inner_corner_y','mouth_center_top_lip_y','nose_tip_y']
    nose_mean_y = facial_keypoints_df.loc[image_number, nose_keypoints_y].mean()
    data['nose_y'].append(nose_mean_y)

    # calculate nose height
    left_nose_height = abs(facial_keypoints_df.loc[image_number, 'left_eye_inner_corner_y'] - 
                           (0.5 * facial_keypoints_df.loc[image_number, 'nose_tip_y']) + 
                           facial_keypoints_df.loc[image_number, 'mouth_center_top_lip_y'])
    right_nose_height = abs(facial_keypoints_df.loc[image_number, 'right_eye_inner_corner_y'] - 
                           (0.5 * facial_keypoints_df.loc[image_number, 'nose_tip_y']) + 
                           facial_keypoints_df.loc[image_number, 'mouth_center_top_lip_y'])
    data['nose_height'].append(0.2 * (left_nose_height + right_nose_height))

    # calculate nose width
    nose_upper_width = abs(facial_keypoints_df.loc[image_number, 'left_eye_inner_corner_x'] - 
                           facial_keypoints_df.loc[image_number, 'right_eye_inner_corner_x'])
    nose_lower_width = abs(facial_keypoints_df.loc[image_number, 'mouth_left_corner_x'] - 
                           facial_keypoints_df.loc[image_number, 'mouth_right_corner_x'])
    data['nose_width'].append(0.5 * (nose_upper_width + nose_lower_width))

    # calculate mouth x
    mouth_keypoints_x = ['mouth_center_top_lip_x','mouth_center_bottom_lip_x','mouth_right_corner_x','mouth_left_corner_x']
    mouth_mean_x = facial_keypoints_df.loc[image_number, mouth_keypoints_x].mean()
    data['mouth_x'].append(mouth_mean_x)

    # calculate mouth y
    mouth_keypoints_y = ['mouth_center_top_lip_y','mouth_center_bottom_lip_y','mouth_right_corner_y','mouth_left_corner_y']
    mouth_mean_y = facial_keypoints_df.loc[image_number, mouth_keypoints_y].mean()
    data['mouth_y'].append(mouth_mean_y)

    # calculate mouth width
    mouth_width = 1.3 * abs(facial_keypoints_df.loc[image_number, 'mouth_left_corner_x'] -
                            facial_keypoints_df.loc[image_number, 'mouth_right_corner_x'])
    data['mouth_width'].append(mouth_width)

    # calculate mouth height
    mouth_height = 7 + 0.95 * abs(facial_keypoints_df.loc[image_number, 'mouth_center_top_lip_y'] -
                            facial_keypoints_df.loc[image_number, 'mouth_center_bottom_lip_y'])
    data['mouth_height'].append(mouth_height)

    return pd.DataFrame(data)

if __name__ == "__main__":
    main()