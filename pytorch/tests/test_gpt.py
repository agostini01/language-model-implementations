import unittest
import torch

from mygpt.gpt import GPT
from mygpt.runner import Solution


class TestSingleHeadAttention(unittest.TestCase):
    def test_single_head_attention(self):
        # Define the input and the expected output
        embedded = torch.tensor([
            [[-1.4381, 0.1232],
             [-0.1080, 0.3458]],
            [[0.1929, -0.8567],
             [-0.1160, 1.2547]]
        ])
        expected_output = torch.tensor([
            [[-0.9737, 0.4302, -0.4216],
             [-2.4031, 1.4092, 1.3797]],
            [[1.7862, -2.1856, 0.2375],
             [-0.7592, -0.1953, -0.4658]]
        ])

        # Define the SingleHeadAttention instance
        single_head_attention = GPT.TransformerBlock.MultiHeadAttention.SingleHeadAttention(
            2, 3)

        # Call the forward method and check the output
        output = single_head_attention(embedded)
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-4))


class TestSolution(unittest.TestCase):
    def test_generate(self):
        # Define the mock model
        class MockModel:
            def __call__(self, context):
                output = torch.tensor([
                    [0.0000, 0.8000, 0.1000, 0.0000, 0.1000],
                    [0.0000, 0.0000, 0.9000, 0.0000, 0.1000],
                    [0.0500, 0.0000, 0.0000, 0.9500, 0.0000],
                    [0.0000, 0.7000, 0.0000, 0.0000, 0.3000],
                    [0.0000, 0.0000, 0.1000, 0.0000, 0.9000]
                ])
                return output.unsqueeze(0).unsqueeze(0)

        # Define the context and the expected output
        # 'With', 'great', 'power', 'comes', 'great'
        context = torch.tensor([[0, 1, 2, 3, 1]])
        expected_output = 'great'

        # Define the Solution instance and the mock model
        solution = Solution()
        model = MockModel()

        # Define the int_to_char mapping
        int_to_char = {0: 'with', 1: 'great',
                       2: 'power', 3: 'comes', 4: 'responsibility'}

        # Call the generate method and check the output
        output = solution.generate(model, 1, context, 5, int_to_char)
        self.assertEqual(output, expected_output)


if __name__ == '__main__':
    unittest.main()
