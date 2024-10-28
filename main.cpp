#include <iostream>
#include "GPUroutines.cuh"
using namespace std;

int main () {
    cout<<"Hello world!!!!!"<<endl;
    GPU sub;
    for (int i=0;i<10;++i){
        sub.writeNum(i);
    }
    return 0;
}
