// Standalone logic self-test for torch::text (vocab + tokenizers only,
// no ATen dependency — pure stdlib, runnable under LCC).
#include "torch/text/tokenizers.h"
#include "torch/text/vocab.h"

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
        assert(v["zzz"] == v.unk_id());
        std::printf("[vocab] round-trip OK (%zu tokens)\n", v.size());
    }

    // --- BPE deterministic IDs --------------------------------------------
    {
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
        assert(toks.size() == 2);
        assert(toks[0] == "Hello");
        assert(toks[1] == "world");
        auto ids = t.encode("Hello world");
        assert(ids.size() == 2);
        std::printf("[bpe] Hello/world merged -> ids (%ld, %ld)\n",
                    (long)ids[0], (long)ids[1]);
    }

    // --- WordPiece "unaffordable" -----------------------------------------
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
        std::printf("[wordpiece] un+##afford+##able OK, decode='%s'\n",
                    dec.c_str());
    }

    // --- CharTokenizer UTF-8 ----------------------------------------------
    {
        std::vector<std::string> toks{"H", "i", "!"};
        Vocab v = Vocab::from_iterator(toks);
        CharTokenizer ct(v);
        auto pieces = ct.tokenize("Hi!");
        assert(pieces.size() == 3);
        assert(pieces[0] == "H" && pieces[1] == "i" && pieces[2] == "!");
        std::printf("[char] Hi! -> 3 codepoints\n");
    }

    std::printf("All torch::text logic self-tests passed.\n");
    return 0;
}
