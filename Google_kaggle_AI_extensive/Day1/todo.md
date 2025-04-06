- Each input embedding is multiplied by three learned
  weight matrices (Wq, Wk, Wv) to generate query (Q), key (K), and value (V) vectors. These
  are like specialized representations of each word.
- example of the following
  Query: The query vector helps the model ask, “Which other words in the sequence are
  relevant to me?”
  • Key: The key vector is like a label that helps the model identify how a word might be
  relevant to other words in the sequence.
  • Value: The value vector holds the actual word content information.
- why softmax ?why not others
- The outputs
  from each head are concatenated and linearly transformed, giving the model a richer
  representation of the input sequence.
- how ?
  The use of multi-head attention improves the model’s ability to handle complex language
  patterns and long-range dependencies.
