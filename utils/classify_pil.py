import logging
import numpy as np
import onnxruntime
from PIL import Image as pil_image


if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        "nearest": pil_image.NEAREST,
        "bilinear": pil_image.BILINEAR,
        "bicubic": pil_image.BICUBIC,
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, "HAMMING"):
        _PIL_INTERPOLATION_METHODS["hamming"] = pil_image.HAMMING
    if hasattr(pil_image, "BOX"):
        _PIL_INTERPOLATION_METHODS["box"] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, "LANCZOS"):
        _PIL_INTERPOLATION_METHODS["lanczos"] = pil_image.LANCZOS


def img_to_array(img, data_format="channels_last", dtype="float32"):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {"channels_first", "channels_last"}:
        raise ValueError("Unknown data_format: %s" % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == "channels_first":
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == "channels_first":
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError("Unsupported image shape: %s" % (x.shape,))
    return x


def load_unsave_images(images, target_size):
    """
    Function for loading images into numpy arrays for passing to model.predict
    inputs:
        images: list of PIL images to load
        image_size: size into which images should be resized
    
    outputs:
        loaded_images: loaded images on which keras model can run predictions
    
    """
    loaded_images = []
    interpolation="nearest"

    for i, image in enumerate(images):
        try:
            # resize image to model input size
            if target_size is not None:
                width_height_tuple = (target_size[1], target_size[0])
                if image.size != width_height_tuple:
                    if interpolation not in _PIL_INTERPOLATION_METHODS:
                        raise ValueError(
                            "Invalid interpolation method {} specified. Supported "
                            "methods are {}".format(
                                interpolation, ", ".join(_PIL_INTERPOLATION_METHODS.keys())
                            )
                        )
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            image = image.resize(width_height_tuple, resample)

            # convert image to array
            image = img_to_array(image)
            image /= 255
            loaded_images.append(image)
            # loaded_image_paths.append(image_names[i])
        except Exception as ex:
            logging.exception(f"Error reading {ex}", exc_info=True)

    return np.asarray(loaded_images)


class Classifier:
    """
    Class for loading model and running predictions.
    For example on how to use take a look the if __name__ == '__main__' part.
    """
    nsfw_model = None

    def __init__(self, model_path):
        """
        model = Classifier()
        """
        # url = "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx"
        # url = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/320n.onnx"
        # home = os.path.expanduser("~")
        # model_folder = os.path.join(home, ".NudeNet/")
        # if not os.path.exists(model_folder):
        #     os.mkdir(model_folder)

        # model_path = os.path.join(model_folder, os.path.basename(url))

        # if not os.path.exists(model_path):
        #     print("Downloading the checkpoint to", model_path)
        #     pydload.dload(url, save_to_path=model_path, max_time=None)

        self.nsfw_model = onnxruntime.InferenceSession(model_path)

    def classify(
        self,
        images=[],
        image_names=[],
        batch_size=4,
        image_size=(256, 256),
        categories=["unsafe", "safe"],
    ):
        """
        inputs:
            image_paths: list of image paths or can be a string too (for single image)
            batch_size: batch_size for running predictions
            image_size: size to which the image needs to be resized
            categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        if not isinstance(images, list):
            images = [images]

        loaded_images = load_unsave_images(
            images, image_size
        )
        loaded_image_paths = image_names

        # print("loaded_images shape:", loaded_images.shape)

        if not loaded_image_paths:
            return {}

        preds = []
        model_preds = []
        while len(loaded_images):
            _model_preds = self.nsfw_model.run(
                [self.nsfw_model.get_outputs()[0].name],
                {self.nsfw_model.get_inputs()[0].name: loaded_images[:batch_size]},
            )[0]
            model_preds.append(_model_preds)
            preds += np.argsort(_model_preds, axis=1).tolist()
            loaded_images = loaded_images[batch_size:]

        # print(preds)
        # print("Raw model output:", _model_preds)
        # print("Shape:", _model_preds.shape)
        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            # if isinstance(single_preds[0], list):
            #     single_preds = single_preds[0]
            for j, pred in enumerate(single_preds):
                single_probs.append(
                    model_preds[int(i / batch_size)][int(i % batch_size)][pred]
                )
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            if not isinstance(loaded_image_path, str):
                loaded_image_path = i

            images_preds[loaded_image_path] = {}
            for _ in range(len(preds[i])):
                images_preds[loaded_image_path][preds[i][_]] = float(probs[i][_])

        return images_preds


# if __name__ == "__main__":
#     m = Classifier()

#     while 1:
#         print(
#             "\n Enter single image path or multiple images seperated by || (2 pipes) \n"
#         )
#         images = input().split("||")
#         images = [image.strip() for image in images]
#         print(m.predict(images), "\n")
