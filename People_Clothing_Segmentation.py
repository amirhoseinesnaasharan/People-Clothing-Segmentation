import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import tqdm
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Input, Conv2DTranspose, Concatenate
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.applications import MobileNetV2  # اطمینان از وارد شدن MobileNetV2
import glob

# مسیر داده‌ها
images_path = r"C:\Users\Amir\Desktop\python\code python\People ClothingSegmentation\People Clothing Segmentation\png_images\IMAGES"
masks_path = r"C:\Users\Amir\Desktop\python\code python\People ClothingSegmentation\People Clothing Segmentation\png_masks\MASKS"

# مسیر ذخیره مدل
model_save_path = r"C:\Users\Amir\Desktop\python\code python\People ClothingSegmentation\People Clothing Segmentation\models"
os.makedirs(model_save_path, exist_ok=True)

# تعداد کانال‌های خروجی
OUTPUT_CHANNELS = 9

# تعریف کلاس‌ها
classes = {'bg':0, 'accessories': 1, 'bag': 2, 'clothes': 3, 'shoes': 4, 'glasses': 5, 'hair': 6, 'hat': 7, 'skin': 8}

# تعریف نگاشت قدیمی به جدید
convs = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 4, 8: 3, 9: 1, 10: 3, 11: 3, 12: 4, 13: 3, 14: 3, 15: 1, 
         16: 4, 17: 5, 18: 3, 19: 6, 20: 7, 21: 4, 22: 3, 23: 3, 24: 3, 25: 3, 26: 3, 27: 3, 28: 4, 29: 1, 
         30: 3, 31: 3, 32: 4, 33: 3, 34: 1, 35: 3, 36: 4, 37: 3, 38: 3, 39: 4, 40: 3, 41: 8, 42: 3, 43: 4, 
         44: 3, 45: 3, 46: 3, 47: 5, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3, 53: 3, 54: 3, 55: 3, 56: 3, 57: 1, 
         58: 4}

# توابع پیش‌پردازش
def standardize(x):
    x = np.array(x, dtype='float64')
    x -= np.min(x)
    x /= np.percentile(x, 98)
    x[x > 1] = 1
    return x

def preprocessing(img):
    image = np.array(img)   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.zeros_like(image)
    image[:,:,0] = gray
    image[:,:,1] = gray
    image[:,:,2] = gray
    image = standardize(image)
    return image

# بارگذاری مسیر تصاویر و ماسک‌ها با استفاده از glob برای فیلتر کردن فایل‌های تصویری
images_paths = glob.glob(os.path.join(images_path, "*.png"))  # فرض بر این است که تصاویر با پسوند .png هستند
masks_paths = glob.glob(os.path.join(masks_path, "*.png"))    # فرض بر این است که ماسک‌ها با پسوند .png هستند

# بررسی تعداد فایل‌های پیدا شده
print(f"تعداد تصاویر پیدا شده: {len(images_paths)}")
print(f"تعداد ماسک‌ها پیدا شده: {len(masks_paths)}")

# اطمینان از تطابق تعداد تصاویر و ماسک‌ها
if len(images_paths) != len(masks_paths):
    print("تعداد تصاویر و ماسک‌ها برابر نیست. لطفاً مسیرها را بررسی کنید.")
    exit(1)

images_paths.sort()
masks_paths.sort()

print("نمونه‌ای از مسیر تصاویر:", images_paths[:5])
print("نمونه‌ای از مسیر ماسک‌ها:", masks_paths[:5])

# اندازه‌گیری تصاویر
SIZE_X = 224 
SIZE_Y = 224
n_classes = 9  # تعداد کلاس‌ها برای تقسیم‌بندی

# بارگذاری و پیش‌پردازش تصاویر و ماسک‌ها
train_images = []
train_masks = [] 

for imgpath in tqdm.tqdm(images_paths, desc="بارگذاری تصاویر"):
    img = cv2.imread(imgpath)
    if img is None:
        print(f"هشدار: نمی‌توانم تصویر را بخوانم - {imgpath}")
        continue
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = preprocessing(img)               
    train_images.append(img)

for maskpath in tqdm.tqdm(masks_paths, desc="بارگذاری ماسک‌ها"):
    mask0 = cv2.imread(maskpath, 0)
    if mask0 is None:
        print(f"هشدار: نمی‌توانم ماسک را بخوانم - {maskpath}")
        continue
    mask1 = cv2.resize(mask0, (SIZE_Y, SIZE_X), interpolation=cv2.INTER_NEAREST)
    for oldnum, new_num in convs.items():
        mask1[mask1 == oldnum] = new_num
    train_masks.append(mask1)

train_images = np.array(train_images)
train_masks = np.array(train_masks)

# بررسی تعداد داده‌های بارگذاری شده
print(f"تعداد تصاویر بارگذاری شده: {len(train_images)}")
print(f"تعداد ماسک‌ها بارگذاری شده: {len(train_masks)}")

# تقسیم داده‌ها به آموزش و اعتبارسنجی
X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.10, shuffle=True, random_state=1)
print("مقادیر کلاس‌ها در داده‌های آموزشی:", np.unique(y_train))

# نرمال‌سازی برای نمایش
NORM = mpl.colors.Normalize(vmin=0, vmax=8)

# نمایش نمونه‌ای از تصاویر و ماسک‌ها
plt.figure(figsize=(16,10))
for i in range(1,4):
    plt.subplot(2,3,i)
    img = train_images[i]
    plt.imshow(img)
    plt.colorbar()
    plt.axis('off')

for i in range(4,7):
    plt.subplot(2,3,i)
    img = np.squeeze(train_masks[i-3])
    plt.imshow(img, cmap='jet', norm=NORM)
    plt.colorbar()
    plt.axis('off')
plt.show()

# تعریف مدل U-Net
def unet_model(output_channels):
    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH  = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]

    base_model = MobileNetV2(input_shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], include_top=False)

    # انتخاب لایه‌های برای اتصال‌های پرش
    layer_names = [
        'block_1_expand_relu',   # 112x112
        'block_3_expand_relu',   # 56x56
        'block_6_expand_relu',   # 28x28
        'block_13_expand_relu',  # 14x14
        'block_16_project',      # 7x7
    ]

    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # ایجاد مدل استخراج ویژگی
    down_stack = Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 7x7 -> 14x14
        pix2pix.upsample(256, 3),  # 14x14 -> 28x28
        pix2pix.upsample(128, 3),  # 28x28 -> 56x56
        pix2pix.upsample(64, 3),   # 56x56 -> 112x112
    ]

    inputs = Input(shape=[IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

    # Downsampling از طریق مدل
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling و اتصال پرش
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = Concatenate()
        x = concat([x, skip])

    # لایه نهایی مدل با فعال‌سازی softmax
    last = Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='softmax')  # 112x112 -> 224x224
    x = last(x)

    return Model(inputs=inputs, outputs=x)

# تعریف متریک سفارشی MeanIoU که ابتدا پیش‌بینی‌ها را به کلاس‌های انتهایی تبدیل می‌کند
class MeanIoUWithArgmax(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# تابع ایجاد ماسک از پیش‌بینی مدل
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# تابع نمایش پیش‌بینی‌ها
def show_predictions(epoch, dataset=None, num=50):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            plt.figure(figsize=(15, 10))
            plt.subplot(231)
            plt.title('تصویر تست')
            plt.imshow(image[0], cmap='gray')
            plt.subplot(232)
            plt.title('Ground Truth')
            plt.imshow(mask[0], cmap='jet')
            plt.subplot(233)
            plt.title('پیش‌بینی مدل')
            plt.imshow(create_mask(pred_mask), cmap='jet')
            plt.axis('off')
            plt.show()
    else:
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle(f"\n Epoch: {str(epoch)}\n", fontsize=16)

        plt.subplot(331)
        plt.title('تصویر تست')
        plt.imshow(train_images[num], cmap='gray')
        plt.subplot(332)
        plt.title('Ground Truth')
        plt.imshow(train_masks[num], cmap='jet')
        plt.subplot(333)
        plt.title('پیش‌بینی مدل')
        plt.imshow(create_mask(model.predict(train_images[num][tf.newaxis, ...]))[:,:,0], cmap='jet')

        plt.subplot(334)
        plt.imshow(train_images[num+16], cmap='gray')
        plt.subplot(335)
        plt.imshow(train_masks[num+16], cmap='jet')
        plt.subplot(336)
        plt.imshow(create_mask(model.predict(train_images[num+16][tf.newaxis, ...]))[:,:,0], cmap='jet')

        plt.subplot(337)
        plt.imshow(train_images[num+14], cmap='gray')
        plt.subplot(338)
        plt.imshow(train_masks[num+14], cmap='jet')
        plt.subplot(339)
        plt.imshow(create_mask(model.predict(train_images[num+14][tf.newaxis, ...]))[:,:,0], cmap='jet')

        plt.show()

# ایجاد و کامپایل مدل
model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=False),  # تغییر from_logits به False
              metrics=['accuracy', MeanIoUWithArgmax(num_classes=OUTPUT_CHANNELS)])

# تنظیمات آموزش
EPOCHS = 50
BATCH_SIZE = 16

# تعریف Callback برای نمایش پیش‌بینی‌ها
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(epoch)
        print(f'\nپیش‌بینی نمونه‌ای پس از اتمام Epoch {epoch+1}\n')

# آموزش مدل
model_history = model.fit(X_train, y_train, epochs=EPOCHS,
                          batch_size=BATCH_SIZE, 
                          verbose=1, 
                          validation_data=(X_val, y_val),
                          callbacks=[DisplayCallback()])

# ذخیره مدل آموزش دیده
model_save_full_path = os.path.join(model_save_path, "clothes_50.h5")
model.save(model_save_full_path)
model = load_model(model_save_full_path, compile=False)

# visualize results
test_path = r"C:\Users\Amir\Desktop\python\code python\People ClothingSegmentation\People Clothing Segmentation\test_images"

test_paths = glob.glob(os.path.join(test_path, "*.png"))  # فرض بر این است که تصاویر تست با پسوند .png هستند

if len(test_paths) == 0:
    print("هیچ فایل تصویری در مسیر تست پیدا نشد.")
    exit(1)

timgnum = 0
try:
    img_num = int(os.path.basename(test_paths[timgnum]).split(".")[0].split("_")[-1])
except ValueError:
    print(f"فرمت نام فایل تست نامعتبر است: {test_paths[timgnum]}")
    exit(1)

plt.figure(figsize=(16,10))

# نمایش تصویر تست
plt.subplot(2,3,1)
img = cv2.imread(test_paths[timgnum])
if img is None:
    print(f"هشدار: نمی‌توانم تصویر تست را بخوانم - {test_paths[timgnum]}")
else:
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    img = preprocessing(img)
    plt.imshow(img, cmap='gray')

# پیش‌بینی و نمایش ماسک
plt.subplot(2,3,2)
if img is not None:
    pred = np.array(create_mask(model.predict(img[tf.newaxis, ...])))
    plt.imshow(np.squeeze(pred), cmap='jet')

# نمایش Ground Truth (توجه: ممکن است نیاز به تطابق شماره تصویر باشد)
if img_num-1 < len(train_masks):
    plt.subplot(2,3,3)
    plt.imshow(train_masks[img_num-1], cmap='jet')
else:
    plt.subplot(2,3,3)
    plt.title('Ground Truth Not Available')

# نمایش نمودار دقت
history_1 = model_history.history
acc = history_1.get('accuracy', [])
val_acc = history_1.get('val_accuracy', [])

plt.figure(figsize=(8,6))
plt.plot(acc, '-', label='Training Accuracy')
plt.plot(val_acc, '--', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0.7,1.0])
plt.legend()
plt.show()
