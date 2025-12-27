import cv2
import numpy as np
import matplotlib.pyplot as plt
def show_image(img_name, image):
    cv2.imshow(f'{img_name}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'{img_name}.jpg', image)

def convovle(image, kernel):
    image_height, image_width, num_channel = image.shape
    kernel_height, kernel_width = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))
    padding_height = kernel_height //2
    padding_width = kernel_width //2
    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width), (0 , 0)), mode='constant')
    output_image = np.zeros_like(image)
    for c in range(num_channel):
        for y in range(image_height):
            for x in range(image_width):
                pixel = np.sum(kernel * padded_image[y:y+kernel_height, x:x+kernel_width, c])
                output_image[y, x, c] = np.clip(pixel, 0, 255)
    return output_image

def apply_spatial_domain_filter(image, image_name, kernel):
    filtered_image = convovle(image, kernel)
    #filtered_image = cv2.filter2D(image, -1, kernel)
    show_image(f'{image_name} Filtered Image', filtered_image)

def apply_frequency_domain_filter(image, image_name):
    k = 1
    image_height, image_width, num_channel = image.shape
    all_channel_back = []
    for c in range(num_channel):
        channel = image[:, :, c]
        denormMax = np.max(channel)
        denormMin = np.min(channel)
        channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) #normalize
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        plt.imshow(np.log(1+np.abs(fshift)), cmap='gray')
        plt.axis('off')
        plt.title(f'{image_name} Frequency Domain')
        # plt.savefig(f'{image_name}_Frequency_Domain.jpg')    
        # plt.show()
        cenRow, cenCol = image_height//2, image_width//2
        filter = np.zeros_like(channel)
        for i in range(image_height):
            for j in range(image_width):
                filter[i, j] = -4 * (np.pi**2) * ((i-cenRow)**2 + (j-cenCol)**2)
        plt.imshow(filter, cmap='gray')
        plt.axis('off')
        plt.title(f'{image_name} Filter')
        # plt.savefig(f'{image_name}_Filter.jpg')
        # plt.show()
        Lap = fshift * filter
        Lap_ishift = np.fft.ifftshift(Lap)
        channel_back = np.fft.ifft2(Lap_ishift).real
        channel_back = channel_back / np.max(channel_back)
        new_channel = channel - k * channel_back
        new_channel = new_channel * (denormMax - denormMin) + denormMin
        new_channel = np.clip(new_channel, 0, 255)
        all_channel_back.append(new_channel)
    
    img_back = np.stack(all_channel_back, axis=-1).astype(np.uint8)
    # show_image(f'{image_name} Filtered Image', img_back)
    return img_back



# img = cv2.imread('Q1.jpg', 0) #Grayscale Mode
c = 1
img = cv2.imread('Q1.jpg', 1)
cross_kernel = np.array([
    [0, -1*c, 0],
    [-1*c, 1+4*c, -1*c],
    [0, -1*c, 0]
])
# cross_kernel = np.array([
#     [0, 0, -1*c, 0, 0],
#     [0, -1*c, -2*c, -1*c, 0],
#     [-1*c, -2*c, 1+16*c, -2*c, -1*c],
#     [0, -1*c, -2*c, -1*c, 0],
#     [0, 0, -1*c, 0, 0]
# ])

around_kernel = np.array([
    [-1*c, -1*c, -1*c],
    [-1*c, 1+8*c, -1*c],
    [-1*c, -1*c, -1*c]
])
# around_kernel = np.array([
#     [-1*c, -1*c, -1*c, -1*c, -1*c],
#     [-1*c, -1*c, -1*c, -1*c, -1*c],
#     [-1*c, -1*c, 1 + 24*c, -1*c, -1*c],
#     [-1*c, -1*c, -1*c, -1*c, -1*c],
#     [-1*c, -1*c, -1*c, -1*c, -1*c],
# ])
apply_spatial_domain_filter(img, 'Cross_Kernel', cross_kernel)
apply_spatial_domain_filter(img, 'Around_Kernel', around_kernel)
new_img = apply_frequency_domain_filter(img, 'Q1')
show_image('New Image', new_img)
# show_image('Original Image', img)
