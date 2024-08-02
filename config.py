
def Config(object):
    OUTPUT_CHANNELS = 1 #Grey channel
    
    # The facade training set consist of 400 images
    BUFFER_SIZE = 1901
    # The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
    BATCH_SIZE = 1
    # Each image is 256x256 in size
    IMG_WIDTH = 128
    IMG_HEIGHT = 128