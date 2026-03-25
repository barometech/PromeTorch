#include <iostream>
#include "nm_quad_load.h"
int main() {
    unsigned int count = 0;
    PL_GetBoardCount(&count);
    std::cout << "Boards: " << count << std::endl;
    
    for (unsigned int b = 0; b < count; b++) {
        PL_Board* board;
        PL_GetBoardDesc(b, &board);
        PL_ResetBoard(board);
        PL_LoadInitCode(board);
        
        // Try all chip/core combinations
        for (int cluster = 0; cluster < 4; cluster++) {
            for (int nm = 0; nm < 4; nm++) {
                PL_CoreNo core = {nm, cluster};
                PL_Access* access = nullptr;
                int ret = PL_GetAccess(board, &core, &access);
                if (ret == PL_OK && access) {
                    std::cout << "  Board " << b << " chip " << cluster << " core " << nm << " OK" << std::endl;
                    PL_CloseAccess(access);
                }
            }
        }
        PL_CloseBoardDesc(board);
    }
    return 0;
}
