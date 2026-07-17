import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def organize_data(base_raw='xray_raw/chest_xray/train', base_out='structured_data'):
    """
    Organizes raw chest X-ray images into a clean structured directory (Normal vs Pneumonia).
    """
    if os.path.exists(base_out):
        print(f"Structured data folder '{base_out}' already exists. Skipping reorganization.")
        return

    class_names = ['NORMAL', 'PNEUMONIA']

    # Create target directories
    os.makedirs(os.path.join(base_out, 'Normal'), exist_ok=True)
    os.makedirs(os.path.join(base_out, 'Pneumonia'), exist_ok=True)

    for cls in class_names:
        src_path = os.path.join(base_raw, cls)
        if not os.path.exists(src_path):
            print(f"Warning: Source path {src_path} does not exist. Skipping.")
            continue

        target_name = 'Normal' if cls == 'NORMAL' else 'Pneumonia'
        dst_path = os.path.join(base_out, target_name)

        for file in os.listdir(src_path):
            file_src = os.path.join(src_path, file)
            file_dst = os.path.join(dst_path, file)
            if os.path.isfile(file_src):
                shutil.copy(file_src, file_dst)

    print(f"Successfully organized raw data into structured directory '{base_out}'.")

def create_generators(base_out='structured_data', img_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Creates augmented training and validation generators from the structured data.
    """
    # Training generator with heavy data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=validation_split
    )

    # Validation generator without augmentation (only rescale)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    # Generators
    train_gen = train_datagen.flow_from_directory(
        base_out,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        base_out,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    return train_gen, val_gen
