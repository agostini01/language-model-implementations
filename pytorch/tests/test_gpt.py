import unittest
import torch

from mygpt import SingleHeadAttention


class TestSingleHeadAttention(unittest.TestCase):

    def setUp(self):
        self.embedding_dim = 128
        self.attention_dim = 64
        self.batch_size = 32
        self.seq_len = 10
        self.single_head_attention = SingleHeadAttention(
            self.embedding_dim, self.attention_dim)

    def test_forward(self):
        # Create a random tensor to represent the embedded input
        embedded = torch.randn(
            self.batch_size, self.seq_len, self.embedding_dim)

        # Run the forward pass
        output = self.single_head_attention.forward(embedded)

        # Check the output shape
        self.assertEqual(output.shape, (self.batch_size,
                         self.seq_len, self.attention_dim))

        # Check the output values are rounded to 4 decimal places
        rounded_output = torch.round(output, decimals=4)
        self.assertTrue(torch.all(output.eq(rounded_output)))


if __name__ == '__main__':
    unittest.main()
