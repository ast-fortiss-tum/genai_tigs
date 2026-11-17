"""GPU check: dptig.tgate_patch gives bit-identical images to upstream tgate (needs CUDA + weights; ~1 min)."""
import unittest

import numpy as np
import torch

from dptig.config import load_config, resolve_weight


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestTgatePatch(unittest.TestCase):
    def test_bit_identical(self):
        cfg = load_config("configs/cifar10.yaml")
        if not resolve_weight(cfg.sd["lora"]).exists():
            self.skipTest("weights/ missing")
        import tgate.SD_DeepCache as sd_dc
        from dptig import tgate_patch
        from dptig.generator import SDGenerator

        upstream = sd_dc.register_tgate_forward
        try:
            sd_dc.register_tgate_forward = upstream
            gen = SDGenerator(cfg.sd, torch.device("cuda"), patch_tgate=False)
            gen.pipe.set_progress_bar_config(disable=True)
            p = "A photo of deer4 cifar10_4"
            z = gen.latent(3)
            zs = torch.cat([z + 0.01 * i for i in range(4)])

            def run():
                return [np.asarray(gen.seed_image(p, z))] + [np.asarray(im) for im in gen.batch(p, zs)]

            ref = run()
            tgate_patch.apply()
            out = run()
            attn = [m for m in gen.pipe.unet.modules() if m.__class__.__name__ == "Attention"]
            self.assertTrue(attn)
            self.assertTrue(all(getattr(m.forward, "__func__", None) is tgate_patch._forward for m in attn))
        finally:
            sd_dc.register_tgate_forward = upstream
        for a, b in zip(ref, out):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
