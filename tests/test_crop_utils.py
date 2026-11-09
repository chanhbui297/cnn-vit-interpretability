"""Phase 1 tests for deterministic adversarial crop geometry."""

from hashlib import sha256
import unittest

from PIL import Image

from experiments.adversarial_cropping.crop_utils import (
    CROP_AREAS,
    CROP_POSITIONS,
    crop_image,
    iter_crop_conditions,
)
from src.data import iter_analysis_images, load_analysis_manifest


class CropUtilsTest(unittest.TestCase):
    def test_fixed_grid_on_synthetic_rgb_image(self):
        image = Image.new("RGB", (100, 80), color=(10, 20, 30))
        results = list(iter_crop_conditions(image))

        self.assertEqual(len(results), len(CROP_AREAS) * len(CROP_POSITIONS))
        self.assertEqual(
            [result.condition for result in results],
            [
                "area_095__center",
                "area_095__top_left",
                "area_095__top_right",
                "area_095__bottom_left",
                "area_095__bottom_right",
                "area_080__center",
                "area_080__top_left",
                "area_080__top_right",
                "area_080__bottom_left",
                "area_080__bottom_right",
                "area_065__center",
                "area_065__top_left",
                "area_065__top_right",
                "area_065__bottom_left",
                "area_065__bottom_right",
                "area_050__center",
                "area_050__top_left",
                "area_050__top_right",
                "area_050__bottom_left",
                "area_050__bottom_right",
            ],
        )
        for result in results:
            left, top, right, bottom = result.box
            self.assertEqual(result.image.mode, "RGB")
            self.assertGreater(right, left)
            self.assertGreater(bottom, top)
            self.assertTrue(0 <= left < right <= image.width)
            self.assertTrue(0 <= top < bottom <= image.height)


    def test_crop_grid_is_deterministic(self):
        image = Image.new("RGB", (137, 91), color=(40, 50, 60))
        first = list(iter_crop_conditions(image))
        second = list(iter_crop_conditions(image))

        self.assertEqual(
            [result.box for result in first], [result.box for result in second]
        )
        self.assertEqual(
            [result.condition for result in first],
            [result.condition for result in second],
        )


    def test_five_real_images_generate_twenty_valid_crops(self):
        records = load_analysis_manifest()
        selected = records[:5]
        seen = 0

        for image, label, image_id in iter_analysis_images(selected):
            self.assertEqual(image.mode, "RGB")
            self.assertTrue(0 <= label < 1000)
            self.assertTrue(image_id)
            results = list(iter_crop_conditions(image))
            self.assertEqual(len(results), 20)
            for result in results:
                left, top, right, bottom = result.box
                self.assertEqual(result.image.mode, "RGB")
                self.assertTrue(0 <= left < right <= image.width)
                self.assertTrue(0 <= top < bottom <= image.height)
                self.assertEqual(result.crop_size, (right - left, bottom - top))
            seen += 1

        self.assertEqual(seen, len(selected))


    def test_crop_pixel_output_is_stable(self):
        image = Image.new("RGB", (64, 64), color=(100, 110, 120))
        first = crop_image(image, "area_080", "center").image.tobytes()
        second = crop_image(image, "area_080", "center").image.tobytes()

        self.assertEqual(sha256(first).digest(), sha256(second).digest())


if __name__ == "__main__":
    unittest.main()
