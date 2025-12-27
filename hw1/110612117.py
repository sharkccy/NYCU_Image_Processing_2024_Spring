import numpy as np
import cv2

def show_image(img_name, image):
    cv2.imshow(f'{img_name}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Clockwise_Rotation_Nearest_Neighbor(image, angle = 30):
    ori_height, ori_width = image.shape[:2]
    center = (ori_width//2, ori_height//2)
    rotation_matrix = np.array([[np.cos(np.deg2rad(-angle)), -np.sin(np.deg2rad(-angle))], 
                                [np.sin(np.deg2rad(-angle)), np.cos(np.deg2rad(-angle))]])
    new_image = np.zeros((ori_height, ori_width, 3), np.uint8)
    for i in range(ori_height):
        for j in range(ori_width):
            new_i, new_j = np.array([i - center[1], j - center[0]]) @ rotation_matrix + center
            if 0 <= new_i < ori_height and 0 <= new_j < ori_width:
                new_image[i, j] = image[min(int(new_i), ori_height - 1), min(int(new_j), ori_width - 1)]          
    cv2.imwrite('Nearest_Neighbor Rotated Image.jpg', new_image)
    return new_image
    
def Clockwise_Rotation_Bilinear(image, angle = 30):
    ori_height, ori_width = image.shape[:2]
    center = (ori_width//2, ori_height//2)
    rotation_matrix = np.array([[np.cos(np.deg2rad(-angle)), -np.sin(np.deg2rad(-angle))], 
                                [np.sin(np.deg2rad(-angle)), np.cos(np.deg2rad(-angle))]])
    new_image = np.zeros((ori_height, ori_width, 3), np.uint8)
    for i in range(ori_height):
        for j in range(ori_width):
            new_i, new_j = np.array([i - center[1], j - center[0]]) @ rotation_matrix + center
            if 0 <= new_i < ori_height and 0 <= new_j < ori_width:
                x, y = new_i, new_j
                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, ori_height-1), min(y1 + 1, ori_width-1)
                for channel in range(3):    
                    new_image[i,j, channel] = (np.array([x2 - x, x - x1]) @ np.array([[image[x1, y1, channel], image[x1, y2, channel]], [image[x2, y1, channel], image[x2, y2, channel]]]) @ np.array([[y2 - y], [y - y1]]))
    cv2.imwrite('Bilinear Rotated Image.jpg', new_image)
    return new_image             

def cubic_interpolation(p0, p1, p2, p3, x):
    p0 = p0.astype(np.float32)
    p1 = p1.astype(np.float32)
    p2 = p2.astype(np.float32)
    p3 = p3.astype(np.float32)

    value = ((-0.5) * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3) * pow(x, 3) + (p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3) * pow(x, 2) + ((-0.5) * p0 + 0.5 * p2) * x + p1
    for i in range(3):
        value[i] = max(0, min(value[i], 255))
    return value.astype(np.uint8)

def Clockwise_Rotation_Bicubic(image, angle = 30):
    ori_height, ori_width = image.shape[:2]
    center = (ori_width//2, ori_height//2)
    rotation_matrix = np.array([[np.cos(np.deg2rad(-angle)), -np.sin(np.deg2rad(-angle))], [np.sin(np.deg2rad(-angle)), np.cos(np.deg2rad(-angle))]])
    new_image = np.zeros((ori_height, ori_width, 3), np.uint8)
    for i in range(ori_height):
        for j in range(ori_width):
            new_i, new_j = np.array([i - center[1], j - center[0]]) @ rotation_matrix + center
            if 0 <= new_i < ori_height and 0 <= new_j < ori_width:
                x, y = new_i, new_j
                x1, y1 = int(x), int(y)
                p0 = cubic_interpolation(image[max(0, x1-1), max(0, y1-1)], image[x1, max(0, y1-1)], image[min(ori_width-1,x1+1), max(0, y1-1)], image[min(ori_width-1,x1+2), max(0, y1-1)], x % 1)
                p1 = cubic_interpolation(image[max(0, x1-1), y1], image[x1, y1], image[min(ori_width-1,x1+1), y1], image[min(ori_width-1,x1+2), y1], x % 1)
                p2 = cubic_interpolation(image[max(0, x1-1), min(ori_height - 1, y1+1)], image[x1, min(ori_height - 1, y1+1)], image[min(ori_width-1, x1+1), min(ori_height - 1, y1+1)], image[min(ori_width-1,x1+2), min(ori_height - 1, y1+1)], x % 1)
                p3 = cubic_interpolation(image[max(0, x1-1), min(ori_height - 1, y1+2)], image[x1, min(ori_height - 1, y1+2)], image[min(ori_width-1, x1+1), min(ori_height - 1, y1+2)], image[min(ori_width-1,x1+2), min(ori_height - 1, y1+2)], x % 1)
                new_image[i, j] = cubic_interpolation(p0, p1, p2, p3, y%1)
    cv2.imwrite('Bicubic Rotated Image.jpg', new_image)
    return new_image
    

def Enlarge_Image_Nearest_Neighbor(image, scale = 2):
    ori_height, ori_width = image.shape[:2]
    # print(f'Original height: {ori_height}, Original width: {ori_width}')
    new_height, new_width = ori_height*scale, ori_width*scale
    new_image = np.zeros((new_height, new_width, 3), np.uint8)
    for i in range(new_height):
        for j in range(new_width):
            new_image[i, j] = image[min(int(i/scale), ori_height - 1), min(int(j/scale), ori_width - 1)]
    cv2.imwrite('Nearest Neighbor Enlarged Image.jpg', new_image)
    return new_image


def Enlarge_Image_Bilinear(image, scale = 2):
    ori_height, ori_width = image.shape[:2]
    new_height, new_width = ori_height*scale, ori_width*scale
    new_image = np.zeros((new_height, new_width, 3), np.uint8)
    for i in range(new_height):
        for j in range(new_width):
                x, y = i/scale, j/scale
                x1, y1 = i//scale, j//scale
                x2, y2 = min(x1 + 1, ori_height-1), min(y1 + 1, ori_width-1)
                for channel in range(3):    
                    new_image[i,j, channel] = (np.array([x2 - x, x - x1]) 
                                               @ np.array([[image[x1, y1, channel], image[x1, y2, channel]],[image[x2, y1, channel], image[x2, y2, channel]]]) 
                                               @ np.array([[y2 - y], [y - y1]]))
    cv2.imwrite('Biliear Enlarged Image.jpg', new_image)
    return new_image
                    # print(np.shape(np.array([x2 - x, x - x1])))
                    # print(np.shape(np.array([[image[x1, y1, channel], image[x1, y2, channel]], [image[x2, y1, channel], image[x2, y2, channel]]])))
                    # print(np.shape(np.array([[y2 - y], [y - y1]])))
                    # print(f'{i, j}')


def Enlarge_Image_Bicubic(image, scale = 2):
    ori_height, ori_width = image.shape[:2]
    new_height, new_width = ori_height*scale, ori_width*scale
    new_image = np.zeros((new_height, new_width, 3), np.uint8)
    for i in range(new_height):
        for j in range(new_width):
                # print(f'{i, j}')
                x, y = i/scale, j/scale
                x1, y1 = int(i//scale), int(j//scale)
                p0 = cubic_interpolation(image[max(0, x1-1), max(0, y1-1)], image[x1, max(0, y1-1)], image[min(ori_width-1,x1+1), max(0, y1-1)], image[min(ori_width-1,x1+2), max(0, y1-1)], x % 1)
                p1 = cubic_interpolation(image[max(0, x1-1), y1], image[x1, y1], image[min(ori_width-1,x1+1), y1], image[min(ori_width-1,x1+2), y1], x % 1)
                p2 = cubic_interpolation(image[max(0, x1-1), min(ori_height - 1, y1+1)], image[x1, min(ori_height - 1, y1+1)], image[min(ori_width-1, x1+1), min(ori_height - 1, y1+1)], image[min(ori_width-1,x1+2), min(ori_height - 1, y1+1)], x % 1)
                p3 = cubic_interpolation(image[max(0, x1-1), min(ori_height - 1, y1+2)], image[x1, min(ori_height - 1, y1+2)], image[min(ori_width-1, x1+1), min(ori_height - 1, y1+2)], image[min(ori_width-1,x1+2), min(ori_height - 1, y1+2)], x % 1)
                new_image[i, j] = cubic_interpolation(p0, p1, p2, p3, y%1)
    cv2.imwrite('Bicubic Enlarged Image.jpg', new_image)
    return new_image

     

image = cv2.imread('building.jpg')
# show_image('Original Image', image)
show_image('Nearest Neighbor Enlarged Image', Enlarge_Image_Nearest_Neighbor(image))
show_image('Bilinear Enlarged Image', Enlarge_Image_Bilinear(image))
show_image('Bicubic Enlarged Image', Enlarge_Image_Bicubic(image))
show_image('rotated image', Clockwise_Rotation_Nearest_Neighbor(image))
show_image('rotated image', Clockwise_Rotation_Bilinear(image))
show_image('rotated image', Clockwise_Rotation_Bicubic(image))
# uncomment the above line to see the result of each function

