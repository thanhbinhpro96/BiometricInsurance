# Imports
import argparse
from time import time
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from wide_resnet import WideResNet

def Wide_Resnet_Model(num_classes):
    # Create a WideResNet
    wide_resnet_model = WideResNet(64)()
    # Load pre-trained weights
    wide_resnet_model.load_weights("./models/weights.18-4.06.hdf5")
    # Remove the last 2 layers
    wide_resnet_model.layers.pop()
    wide_resnet_model.layers.pop()
    # Freeze the first 45 layers weights
    for layer in wide_resnet_model.layers[:45]:
        layer.trainable = False
    # Uncomment to print out network structure
    #wide_resnet_model.summary()
    # Create a new classification layer
    x_newfc = Dense(num_classes, activation='softmax', name='classification')(wide_resnet_model.layers[-1].output)
    # Attach the new layer to network
    new_model = keras.Model(inputs=wide_resnet_model.input, outputs=x_newfc)
    # Compile new network
    adam = Adam()
    new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', 'crossentropy'])
    # Save weights
    new_model.save("./models/temp_weights.hdf5")

    return new_model

def get_args():
    # Get arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", type=str)
    parser.add_argument("--valid", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--classes", type=int)
    args = parser.parse_args()
    return args

def train(model, img_cols, img_rows, batch_size, train_data_dir, validation_data_dir, tensorboard, nb_epoch, model_path):
    # Create checkpoint to monitor training process and save best checkpoints
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

    # Load pre-trained weights
    model.load_weights("./models/temp_weights.hdf5")

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=True)

    # Create training batches
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_cols, img_rows),
        batch_size=batch_size,
        class_mode='categorical')

    # Create testing batches
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_cols, img_rows),
        batch_size=batch_size,
        class_mode='categorical')

    print("\n" + "---------------------------------------"
          + "\n" +"Training process commencing..."
          +"\n"+ "---------------------------------------")

    # Start fine-tuning the model
    model.fit_generator(
        train_generator,
        callbacks=[checkpoint],
        verbose=1,
        steps_per_epoch=512,
        epochs=nb_epoch,
        validation_data=validation_generator,
        shuffle=True,
        validation_steps=50)

    # Save weights when done training
    model.save_weights('./models/best_weights.hdf5')

def main():
    args = get_args()
    train_data_dir = args.train
    validation_data_dir = args.valid
    tensorboard = keras.callbacks.TensorBoard(log_dir="./tmp/wide_resnet_log".format(time()))
    img_rows, img_cols = 64, 64  # Resolution of inputs
    num_classes = args.classes
    batch_size = 16
    nb_epoch = 1000
    fine_tune_model = Wide_Resnet_Model(num_classes) # Create a neural network with num_classes output classes
    train(fine_tune_model, img_cols, img_rows, batch_size, train_data_dir, validation_data_dir, tensorboard, nb_epoch, args.model)

if __name__ == '__main__':
    main()