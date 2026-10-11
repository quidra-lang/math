#include "../native/math_exact.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

// Interval invariants are checked directly, independently of Core's proof
// engine. Dyadic rounding covers both signs and carries across limb boundaries.
int main() {
    using namespace math_exact;
    assert(Integer::decimal("18446744073709551616").decimal() == "18446744073709551616");
    assert(floor_div(Integer(-5),Integer(2)) == Integer(-3));
    assert(ceil_div(Integer(-5),Integer(2)) == Integer(-2));
    assert(floor_div(Integer(5),Integer(-2)) == Integer(-3));
    assert(ceil_div(Integer(5),Integer(-2)) == Integer(-2));
    const auto wide=Integer::decimal("340282366920938463463374607431768211456");
    assert(square_root(wide)*square_root(wide)==wide);
    Arithmetic a(160);
    auto pi=a.sub(a.scale(a.atan_reciprocal(5),16),a.scale(a.atan_reciprocal(239),4));
    // Proven decimal rational brackets, not values obtained from this provider.
    auto pi_low=a.fraction(314159265,100000000),pi_high=a.fraction(314159266,100000000);
    assert(pi_low.high<pi.low && pi.high<pi_high.low);
    auto e=a.exp(a.integer(1));
    assert(a.fraction(271828182,100000000).high<e.low);
    assert(e.high<a.fraction(271828183,100000000).low);
    auto log2=a.log(a.integer(2));
    assert(a.fraction(69314718,100000000).high<log2.low);
    assert(log2.high<a.fraction(69314719,100000000).low);
    auto sin1=a.trig(a.integer(1),false);
    assert(a.fraction(84147098,100000000).high<sin1.low);
    assert(sin1.high<a.fraction(84147099,100000000).low);
    qcore_exact_interval out{};
    assert(refine(1,nullptr,0,4097,&out)==0);
    qcore_exact_interval pole{"0","1",0};
    assert(refine(261,&pole,1,80,&out)==0);
    assert(qcore_exact_real_provider_register_v2(&provider)==1);
    Owned seventeen(qcore_exact_real_rational("17","1"));
    const void* argument=seventeen.value;
    Owned sine(qcore_exact_real_node("math",258,&argument,1));
    argument=sine.value;
    Owned logarithm(qcore_exact_real_node("math",261,&argument,1));
    argument=logarithm.value;
    Owned exponential(qcore_exact_real_node("math",262,&argument,1));
    // The sign is outside the refiner's range budget: preserve log's domain.
    assert(qcore_exact_real_is_node(exponential.value,"math",262)==1);
    return 0;
}
