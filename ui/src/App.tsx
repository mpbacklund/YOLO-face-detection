import { useState } from 'react';
import './App.css';
import axios from 'axios';

function App() {
  const [selectedImage, setSelectedImage] = useState<File | Blob | null>(null);

  async function getFaceAugmentations() {
    if (!selectedImage) {
      console.error('No image selected');
      return;
    }

    try {
      // Create a FormData object
      const formData = new FormData();
      formData.append('image', selectedImage); // 'image' is the key your backend expects

      // Make the POST request
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob', // Ensures the response is treated as binary data
      });

      // Save the processed image in the selectedImage state
      const imageBlob = new Blob([response.data], { type: 'image/jpeg' }); // Specify the type for consistency
      setSelectedImage(imageBlob);

      console.log('Image processed and saved');
    } catch (error) {
      console.error('Error sending or receiving the image:', error);
    }
  }

  return (
    <div>
      {/* Header */}
      <h2>Upload and Augment Face Image</h2>

      {/* Render the selected image if it exists */}
      {selectedImage && (
        <div>
          {/* Display the selected or processed image */}
          <img
            alt="Preview"
            width="500px"
            src={selectedImage ? URL.createObjectURL(selectedImage) : ''}
          />
        </div>
      )}

      <br />

      {/* Input element to select an image file */}
      <input
        type="file"
        name="myImage"
        accept="image/*"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            console.log(file); // Log the selected file
            setSelectedImage(file); // Update the state with the selected file
          }
        }}
      />
      <br />

      <button onClick={getFaceAugmentations}>Process Image</button>
    </div>
  );
}

export default App;
