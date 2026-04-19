// Self-test for torch/text.
// Compiles against torch_extras (header-only) and asserts the three
// scenarios spelled out in the task description.
//
//   - BPE: encode "Hello world" with synthetic merges -> deterministic IDs.
//   - Vocab: round-trip encode/decode of sample sentences.
//   - WordPiece: encode "unaffordable" with vocab containing "un", "##afford",
//     "##able" -> 3 IDs.
//
// Run:
//   g++ -std=c++17 -I. torch/text/test_text_selftest.cpp -o text_selftest
//   ./text_selftest
#include "torch/text/text.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    using namespace torch::text;

    // --- Vocab round-trip -------------------------------------------------
    {
        std::vector<std::string> toks{"the", "quick", "brown", "fox"};
        Vocab v = Vocab::from_iterator(toks);
        auto ids = v.encode("the fox the brown");
        assert(ids.size() == 4);
        std::string back = v.decode(ids);
        assert(back == "the fox the brown");
        // Unknown token maps to <unk>.
        assert(v["zzz"] == v.unk_id());
        std::printf("[vocab] round-trip OK (%zu tokens)\n", v.size());
    }

    // --- BPE deterministic IDs --------------------------------------------
    {
        // Build a tiny vocab that covers every character of "Hello world"
        // plus a few post-merge tokens.
        std::vector<std::string> chars{
            "H","e","l","o","w","r","d"," ",
            "He","Hel","Hell","Hello",
            "wo","wor","worl","world"
        };
        Vocab v = Vocab::from_iterator(chars);
        std::vector<std::pair<std::string,std::string>> merges{
            {"H","e"}, {"He","l"}, {"Hel","l"}, {"Hell","o"},
            {"w","o"}, {"wo","r"}, {"wor","l"}, {"worl","d"}
        };
        BPETokenizer t(v, merges);
        auto toks = t.tokenize("Hello world");
        // After greedy merge we expect exactly two pieces.
        assert(toks.size() == 2);
        assert(toks[0] == "Hello");
        assert(toks[1] == "world");
        auto ids = t.encode("Hello world");
        assert(ids.size() == 2);
        assert(ids[0] == v["Hello"]);
        assert(ids[1] == v["world"]);
        std::printf("[bpe] deterministic IDs OK (%ld, %ld)\n",
                    (long)ids[0], (long)ids[1]);
    }

    // --- WordPiece "unaffordable" -> un + ##afford + ##able ---------------
    {
        std::vector<std::string> toks{"un", "##afford", "##able"};
        Vocab v = Vocab::from_iterator(toks);
        WordPieceTokenizer wp(v);
        auto pieces = wp.tokenize("unaffordable");
        assert(pieces.size() == 3);
        assert(pieces[0] == "un");
        assert(pieces[1] == "##afford");
        assert(pieces[2] == "##able");
        auto ids = wp.encode("unaffordable");
        assert(ids.size() == 3);
        std::string dec = wp.decode(ids);
        assert(dec == "unaffordable");
        std::printf("[wordpiece] un+##afford+##able -> 3 IDs, decode='%s'\n",
                    dec.c_str());
    }

    std::printf("All torch::text self-tests passed.\n");
    return 0;
}
