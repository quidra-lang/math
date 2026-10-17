#include "../native/math_exact.hpp"
#include "quidra/abi/exact_codes.hpp"
#include "quidra/abi/runtime_entry_points.hpp"
#include "quidra/abi/layout.hpp"
#include <cstdio>
#include <string>
struct Owned {void* value;explicit Owned(void*p):value(p){}~Owned(){qcore_exact_real_release(value);}};
int main(){
    if(qcore_exact_real_provider_register_v2(&math_exact::provider)!=1)return 2;
    Owned pos(qcore_exact_real_infinity(1)),neg(qcore_exact_real_infinity(-1));
    Owned zero(qcore_exact_real_rational("0","1")),one(qcore_exact_real_rational("1","1"));
    int failures=0;
    auto unary=[&](unsigned op,void* value,const char* expected){
        const void* args[]={value};Owned result(qcore_exact_real_node("math",op,args,1));
        char* str=quidra_bigreal_text(result.value,quidra::abi::exact_real_canonical_text);
        if(std::string(str)!=expected){std::fprintf(stderr,"op %u: got %s wanted %s\n",op,str,expected);++failures;}
        quidra_managed_release(str,nullptr);
    };
    unary(257,pos.value,"∞");unary(261,pos.value,"∞");
    unary(261,zero.value,"-∞");unary(262,pos.value,"∞");unary(262,neg.value,"0.0");
    unary(257,zero.value,"0.0");unary(261,one.value,"0.0");unary(262,zero.value,"1.0");
    std::fprintf(stderr,"%d Math infinity provider failures\n",failures);return failures?1:0;
}
