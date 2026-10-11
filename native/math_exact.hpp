#pragma once

#include <quidra/native_extension.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace math_exact {

// Signed arbitrary integers used only by certified exact-real hooks. Every
// interval operation rounds its lower bound down and its upper bound up.
class Integer {
    std::vector<std::uint32_t> words;
    bool negative{};
    void normalize() {
        while (!words.empty() && words.back() == 0) words.pop_back();
        if (words.empty()) negative = false;
    }
    static int magnitude(const Integer& a, const Integer& b) {
        if (a.words.size() != b.words.size()) return a.words.size() < b.words.size() ? -1 : 1;
        for (std::size_t i = a.words.size(); i != 0; --i)
            if (a.words[i-1] != b.words[i-1]) return a.words[i-1] < b.words[i-1] ? -1 : 1;
        return 0;
    }
    static Integer add_magnitude(const Integer& a, const Integer& b) {
        Integer r; std::uint64_t carry = 0;
        const auto size = std::max(a.words.size(), b.words.size());
        for (std::size_t i = 0; i < size; ++i) {
            carry += (i < a.words.size() ? a.words[i] : 0ULL) +
                     (i < b.words.size() ? b.words[i] : 0ULL);
            r.words.push_back(static_cast<std::uint32_t>(carry)); carry >>= 32;
        }
        if (carry) r.words.push_back(static_cast<std::uint32_t>(carry));
        return r;
    }
    static Integer subtract_magnitude(const Integer& a, const Integer& b) {
        Integer r; std::uint64_t borrow = 0;
        for (std::size_t i = 0; i < a.words.size(); ++i) {
            const std::uint64_t sub = (i < b.words.size() ? b.words[i] : 0ULL) + borrow;
            r.words.push_back(static_cast<std::uint32_t>(std::uint64_t(a.words[i]) - sub));
            borrow = std::uint64_t(a.words[i]) < sub;
        }
        r.normalize(); return r;
    }
public:
    Integer() = default;
    Integer(std::int64_t value) {
        negative = value < 0;
        std::uint64_t n = negative ? std::uint64_t(-(value+1)) + 1 : std::uint64_t(value);
        if (n) words.push_back(static_cast<std::uint32_t>(n));
        if (n >> 32) words.push_back(static_cast<std::uint32_t>(n >> 32));
    }
    static Integer decimal(const char* text) {
        if (!text || !*text) throw std::invalid_argument("empty integer");
        bool neg = *text == '-'; if (neg) ++text;
        if (!*text) throw std::invalid_argument("empty integer");
        Integer n; std::size_t digits = 0;
        for (; *text; ++text) {
            if (*text < '0' || *text > '9' || ++digits > 20000)
                throw std::invalid_argument("invalid integer");
            n = n * Integer(10) + Integer(*text-'0');
        }
        if (neg) n = -n;
        return n;
    }
    std::size_t bits() const {
        if (words.empty()) return 0;
        auto top = words.back(); std::size_t n = 32 * (words.size()-1);
        while (top) { ++n; top >>= 1; } return n;
    }
    bool bit(std::size_t i) const { return i/32 < words.size() && ((words[i/32] >> (i%32)) & 1); }
    Integer abs() const { auto r = *this; r.negative = false; return r; }
    Integer operator-() const { auto r = *this; if (!r.words.empty()) r.negative = !r.negative; return r; }
    Integer shifted(std::size_t n) const {
        if (n > 65536) throw std::length_error("exact shift budget");
        if (words.empty()) return {};
        Integer r; r.negative = negative; r.words.assign(n/32, 0); std::uint64_t carry = 0;
        for (auto word : words) {
            carry |= std::uint64_t(word) << (n%32);
            r.words.push_back(static_cast<std::uint32_t>(carry)); carry >>= 32;
        }
        if (carry) r.words.push_back(static_cast<std::uint32_t>(carry));
        return r;
    }
    friend bool operator<(const Integer& a, const Integer& b) {
        if (a.negative != b.negative) return a.negative;
        return a.negative ? magnitude(a,b) > 0 : magnitude(a,b) < 0;
    }
    friend bool operator==(const Integer& a, const Integer& b) { return a.negative == b.negative && a.words == b.words; }
    friend Integer operator+(const Integer& a, const Integer& b) {
        if (a.negative == b.negative) { auto r = add_magnitude(a,b); r.negative = a.negative; return r; }
        const int c = magnitude(a,b); if (!c) return {};
        auto r = c > 0 ? subtract_magnitude(a,b) : subtract_magnitude(b,a);
        r.negative = c > 0 ? a.negative : b.negative; return r;
    }
    friend Integer operator-(const Integer& a, const Integer& b) { return a + -b; }
    friend Integer operator*(const Integer& a, const Integer& b) {
        Integer r; r.words.assign(a.words.size()+b.words.size(),0);
        for (std::size_t i = 0; i < a.words.size(); ++i) {
            std::uint64_t carry = 0;
            for (std::size_t j = 0; j < b.words.size(); ++j) {
                const std::uint64_t v = std::uint64_t(a.words[i])*b.words[j] + r.words[i+j] + carry;
                r.words[i+j] = static_cast<std::uint32_t>(v); carry = v >> 32;
            }
            if (!b.words.empty()) r.words[i+b.words.size()] = static_cast<std::uint32_t>(carry);
        }
        r.negative = a.negative != b.negative; r.normalize(); return r;
    }
    static Integer quotient(const Integer& a, const Integer& b, bool upward) {
        if (b == Integer()) throw std::domain_error("zero denominator");
        Integer q, rem; const auto divisor = b.abs(); const auto value = a.abs();
        q.words.assign((value.bits()+31)/32,0);
        for (std::size_t i = value.bits(); i != 0; --i) {
            rem = rem.shifted(1) + Integer(value.bit(i-1) ? 1 : 0);
            if (!(rem < divisor)) { rem = subtract_magnitude(rem,divisor); q.words[(i-1)/32] |= std::uint32_t(1) << ((i-1)%32); }
        }
        q.negative = a.negative != b.negative; q.normalize();
        if (!(rem == Integer())) {
            if (upward && !q.negative && !(a.negative != b.negative)) q = q + Integer(1);
            if (!upward && (a.negative != b.negative)) q = q - Integer(1);
        }
        return q;
    }
    std::string decimal() const {
        if (words.empty()) return "0";
        auto value = words; std::string text;
        while (!value.empty()) {
            std::uint64_t rem = 0;
            for (std::size_t i = value.size(); i != 0; --i) {
                const auto v = (rem << 32) | value[i-1]; value[i-1] = static_cast<std::uint32_t>(v/10); rem = v%10;
            }
            text.push_back(static_cast<char>('0'+rem)); while (!value.empty() && !value.back()) value.pop_back();
        }
        if (negative) text.push_back('-');
        std::reverse(text.begin(),text.end());
        return text;
    }
};

inline Integer floor_div(const Integer& a,const Integer& b) { return Integer::quotient(a,b,false); }
inline Integer ceil_div(const Integer& a,const Integer& b) { return Integer::quotient(a,b,true); }
struct Interval { Integer low, high; };
class Arithmetic {
public:
    std::uint32_t precision;
    Integer unit;
    explicit Arithmetic(std::uint32_t bits) : precision(bits), unit(Integer(1).shifted(bits)) {}
    Interval integer(std::int64_t n) const { auto v = Integer(n)*unit; return {v,v}; }
    Interval fraction(std::int64_t n,std::int64_t d) const { return {floor_div(Integer(n)*unit,Integer(d)),ceil_div(Integer(n)*unit,Integer(d))}; }
    Interval add(const Interval& a,const Interval& b) const { return {a.low+b.low,a.high+b.high}; }
    Interval neg(const Interval& a) const { return {-a.high,-a.low}; }
    Interval sub(const Interval& a,const Interval& b) const { return add(a,neg(b)); }
    Interval mul(const Interval& a,const Interval& b) const {
        std::vector<Integer> p{a.low*b.low,a.low*b.high,a.high*b.low,a.high*b.high};
        return {floor_div(*std::min_element(p.begin(),p.end()),unit),ceil_div(*std::max_element(p.begin(),p.end()),unit)};
    }
    Interval div(const Interval& a,const Interval& b) const {
        if (!(b.high < Integer()) && !(Integer() < b.low)) throw std::domain_error("unproved denominator");
        std::vector<Integer> lo,hi;
        for (auto x : {a.low,a.high}) for (auto y : {b.low,b.high}) { lo.push_back(floor_div(x*unit,y)); hi.push_back(ceil_div(x*unit,y)); }
        return {*std::min_element(lo.begin(),lo.end()),*std::max_element(hi.begin(),hi.end())};
    }
    Interval scale(const Interval& a,std::int64_t n) const { return mul(a,integer(n)); }
    Interval import(const qcore_exact_interval& a) const {
        if (a.exponent < -65536 || a.exponent > 65536) throw std::length_error("exact exponent budget");
        auto lo = Integer::decimal(a.low), hi = Integer::decimal(a.high);
        if (hi < lo) throw std::domain_error("reversed interval");
        const auto shift = a.exponent + precision;
        if (shift >= 0) return {lo.shifted(static_cast<std::size_t>(shift)),hi.shifted(static_cast<std::size_t>(shift))};
        auto denominator = Integer(1).shifted(static_cast<std::size_t>(-shift));
        return {floor_div(lo,denominator),ceil_div(hi,denominator)};
    }
    Interval widen(const Interval& x,const Integer& radius) const { return {x.low-radius,x.high+radius}; }
    Interval exp_positive(Interval x) const {
        if (x.low < Integer() || Integer(4096)*unit < x.high) throw std::length_error("exp range budget");
        unsigned squares = 0;
        while (unit < x.high*Integer(2)) { x=div(x,integer(2)); ++squares; }
        Interval sum=integer(1),term=sum;
        for (std::uint32_t n=1;n<=precision+64;++n) {
            term=div(mul(term,x),integer(n));sum=add(sum,term);
            auto next=div(mul(term,x),integer(n+1));
            // x <= 1/2: the remaining positive tail is at most 2*next.
            if (!(Integer(1)<next.high)) {
                sum.high=sum.high+next.high*Integer(2);
                for (unsigned i=0;i<squares;++i) sum=mul(sum,sum);
                return sum;
            }
        }
        throw std::length_error("exp series budget");
    }
    Interval exp(const Interval& x) const {
        auto point=[&](const Integer& value) {
            if (value < Integer()) return div(integer(1),exp_positive({-value,-value}));
            return exp_positive({value,value});
        };
        return {point(x.low).low,point(x.high).high};
    }
    Interval atanh(const Interval& x) const {
        if (x.low < Integer() || unit < x.high*Integer(2)) throw std::domain_error("atanh range");
        const auto square=mul(x,x);auto term=x,sum=x;
        for (std::uint32_t n=1;n<=precision+64;++n) {
            term=mul(term,square);auto next=div(term,integer(2*n+1));sum=add(sum,next);
            const auto tail=div(mul(term,square),integer(2*n+3));
            // x^2 <= 1/4: omitted positive terms are bounded by 2*tail.
            if (!(Integer(1)<tail.high)) { sum.high=sum.high+tail.high*Integer(2);return sum; }
        }
        throw std::length_error("log series budget");
    }
    Interval log_point(Integer x) const {
        if (!(Integer()<x)) throw std::domain_error("log domain");
        const auto original=x; std::int64_t exponent=0;
        while (x < unit) { x=x*Integer(2);--exponent;if (exponent < -65536) throw std::length_error("log range budget"); }
        // Use a dyadic divisor, preserving both outward bounds of normalization.
        while (!(x < unit*Integer(2))) { x=floor_div(x,Integer(2));++exponent;if (exponent > 65536) throw std::length_error("log range budget"); }
        Interval normalized;
        if (exponent>=0) { auto d=Integer(1).shifted(static_cast<std::size_t>(exponent));normalized={floor_div(original,d),ceil_div(original,d)}; }
        else { auto v=original.shifted(static_cast<std::size_t>(-exponent));normalized={v,v}; }
        auto z=div(sub(normalized,integer(1)),add(normalized,integer(1)));
        auto l=scale(atanh(z),2);auto ln2=scale(atanh(fraction(1,3)),2);
        return add(l,scale(ln2,exponent));
    }
    Interval log(const Interval& x) const { return {log_point(x.low).low,log_point(x.high).high}; }
    Interval trig(const Interval& x,bool cosine) const {
        if (x.low < -Integer(16)*unit || Integer(16)*unit < x.high) throw std::length_error("trigonometric range budget");
        auto term=cosine?integer(1):x,sum=term;auto square=mul(x,x);
        for (std::uint32_t n=1;n<=precision+512;++n) {
            const std::int64_t first=cosine?2*n-1:2*n;
            term=neg(div(mul(term,square),integer(first*(first+1))));sum=add(sum,term);
            const auto bound=std::max(term.low.abs(),term.high.abs());
            // Once terms decrease by <=1/2, Taylor's omitted tail is <=2*|term|.
            if (n>=16 && !(Integer(2)<bound)) return widen(sum,bound*Integer(2));
        }
        throw std::length_error("trigonometric series budget");
    }
    Interval atan_reciprocal(std::int64_t d) const {
        auto x=fraction(1,d),square=mul(x,x),term=x,sum=x;
        for (std::uint32_t n=1;n<=precision+64;++n) {
            term=mul(term,square);auto part=div(term,integer(2*n+1));sum=(n%2)?sub(sum,part):add(sum,part);
            const auto bound=part.high;
            if (!(Integer(1)<bound)) return widen(sum,bound*Integer(2));
        }
        throw std::length_error("atan series budget");
    }
};

inline const char* spelling(std::uint32_t op) {
    switch(op) { case 1:return "\xCF\x80";case 2:return "e";case 257:return "sqrt";case 258:return "sin";case 259:return "cos";case 260:return "tan";case 261:return "log";case 262:return "exp";case 263:return "pow";default:return nullptr; }
}
inline std::uint32_t arity(std::uint32_t op) { if(op==1||op==2)return 0;if(op>=257&&op<=262)return 1;if(op==263)return 2;return std::numeric_limits<std::uint32_t>::max(); }
inline std::uint32_t flags(std::uint32_t op) {
    switch(op) {
        case 1:case 2:return QCORE_EXACT_UNARY_RESULT_POSITIVE;
        case 257:return QCORE_EXACT_UNARY_DOMAIN_NONNEGATIVE|QCORE_EXACT_UNARY_RESULT_NONNEGATIVE;
        case 258:case 259:return QCORE_EXACT_UNARY_TOTAL;
        case 261:return QCORE_EXACT_UNARY_DOMAIN_POSITIVE;
        case 262:return QCORE_EXACT_UNARY_TOTAL|QCORE_EXACT_UNARY_RESULT_POSITIVE;
        default:return 0;
    }
}
inline Integer square_root(const Integer& n) {
    if (n < Integer()) throw std::domain_error("negative square root");
    if (n == Integer()) return {};
    auto x=Integer(1).shifted((n.bits()+1)/2);
    for (;;) {
        auto y=floor_div(x+floor_div(n,x),Integer(2));
        if (!(y<x)) return x;
        x=y;
    }
}
struct Owned {
    void* value{};
    explicit Owned(void* v):value(v){}
    ~Owned(){if(value)qcore_exact_real_release(value);}
    Owned(const Owned&)=delete;
    Owned& operator=(const Owned&)=delete;
};
inline void* canonicalize_impl(std::uint32_t op,const void* const* args,std::uint32_t count) {
    if (!args) return nullptr;
    if(op==263&&count==2) {
        // Reuse Core's domain checks. Its generic power need not refine here.
        Owned checked(qcore_exact_real_pow(args[0],args[1]));
        const char *num=nullptr,*den=nullptr;
        if(qcore_exact_real_rational_parts(args[1],&num,&den)) {
            const auto power=Integer::decimal(num),degree=Integer::decimal(den);
            if(power==Integer(1)&&degree==Integer(2))
                return canonicalize_impl(257,args,1);
            if(!(Integer(64)<degree)) return qcore_exact_real_retain(checked.value);
        }
        return nullptr;
    }
    if(count!=1)return nullptr;
    const char *numerator=nullptr,*denominator=nullptr;
    const bool rational=qcore_exact_real_rational_parts(args[0],&numerator,&denominator)!=0;
    Integer n,d;
    if(rational){n=Integer::decimal(numerator);d=Integer::decimal(denominator);}
    if(rational&&n==Integer()) {
        if(op==257||op==258||op==260)return qcore_exact_real_rational("0","1");
        if(op==259||op==262)return qcore_exact_real_rational("1","1");
    }
    if(rational&&n==d&&op==261)return qcore_exact_real_rational("0","1");
    if(op==257) {
        Owned half(qcore_exact_real_rational("1","2"));
        if(rational&&!(n<Integer())) {
            auto residual=n*d;auto coefficient=square_root(residual);
            if(coefficient*coefficient==residual) {
                auto ns=coefficient.decimal(),ds=d.decimal();return qcore_exact_real_rational(ns.c_str(),ds.c_str());
            }
            coefficient=Integer(1);
            // Extract small square factors with a finite, operand-independent budget.
            for(std::int64_t factor=2;factor<=4096;++factor) {
                auto square=Integer(factor)*Integer(factor);
                if(residual<square)break;
                for (;;) { auto q=floor_div(residual,square);if(!(q*square==residual))break;residual=q;coefficient=coefficient*Integer(factor); }
            }
            auto radicand=residual.decimal(),ns=coefficient.decimal(),ds=d.decimal();
            Owned base(qcore_exact_real_rational(radicand.c_str(),"1"));
            Owned root(qcore_exact_real_pow(base.value,half.value));
            Owned scale(qcore_exact_real_rational(ns.c_str(),ds.c_str()));
            return qcore_exact_real_mul(scale.value,root.value);
        }
        return qcore_exact_real_pow(args[0],half.value);
    }
    if (op==261 && qcore_exact_real_is_node(args[0],"math",262)) return qcore_exact_real_argument(args[0],0);
    if (op==262 && qcore_exact_real_is_node(args[0],"math",261)) {
        Owned argument(qcore_exact_real_argument(args[0],0));
        const char *num=nullptr,*den=nullptr;
        // Removing log must not erase a domain check whose input sign is unknown.
        const bool positive=(qcore_exact_real_rational_parts(argument.value,&num,&den) &&
                             Integer()<Integer::decimal(num)) ||
                            qcore_exact_real_is_node(argument.value,"math",1) ||
                            qcore_exact_real_is_node(argument.value,"math",2) ||
                            qcore_exact_real_is_node(argument.value,"math",262);
        if (positive) return qcore_exact_real_retain(argument.value);
    }
    if ((op==258||op==260) && qcore_exact_real_is_node(args[0],"math",1)) return qcore_exact_real_rational("0","1");
    if (op==259 && qcore_exact_real_is_node(args[0],"math",1)) return qcore_exact_real_rational("-1","1");
    return nullptr;
}
inline void* canonicalize(std::uint32_t op,const void* const* args,std::uint32_t count) {
    try { return canonicalize_impl(op,args,count); }
    catch (const std::invalid_argument&) { return nullptr; }
    catch (const std::length_error&) { return nullptr; }
    catch (const std::bad_alloc&) { return nullptr; }
}
inline int prove(std::uint32_t op,const void* const* args,std::uint32_t count,std::uint32_t query) {
    if ((count==0&&(op==1||op==2)) || (count==1&&op==262)) {
        if(query==QCORE_EXACT_QUERY_POSITIVE)return QCORE_EXACT_YES;
        if(query==QCORE_EXACT_QUERY_ZERO||query==QCORE_EXACT_QUERY_NEGATIVE)return QCORE_EXACT_NO;
        if(count==0&&query==QCORE_EXACT_QUERY_INTEGRAL)return QCORE_EXACT_NO;
    }
    if(count!=1||!args)return QCORE_EXACT_UNKNOWN;
    const char *num=nullptr,*den=nullptr;
    if(!qcore_exact_real_rational_parts(args[0],&num,&den))return QCORE_EXACT_UNKNOWN;
    try {
        const auto n=Integer::decimal(num),d=Integer::decimal(den);
        auto answer=[](bool yes){return yes?QCORE_EXACT_YES:QCORE_EXACT_NO;};
        if(op==257&&!(n<Integer())) {
            if(query==QCORE_EXACT_QUERY_ZERO)return answer(n==Integer());
            if(query==QCORE_EXACT_QUERY_POSITIVE)return answer(Integer()<n);
            if(query==QCORE_EXACT_QUERY_NEGATIVE)return QCORE_EXACT_NO;
            if(query==QCORE_EXACT_QUERY_INTEGRAL) {
                auto radicand=n*d,root=square_root(radicand);
                return answer(root*root==radicand && floor_div(root,d)*d==root);
            }
        }
        const bool zero=(n==Integer()&&(op==258||op==260))||(n==d&&op==261);
        const bool one=n==Integer()&&(op==259||op==262);
        if(zero||one) {
            if(query==QCORE_EXACT_QUERY_ZERO)return answer(zero);
            if(query==QCORE_EXACT_QUERY_POSITIVE)return answer(one);
            if(query==QCORE_EXACT_QUERY_NEGATIVE)return QCORE_EXACT_NO;
            if(query==QCORE_EXACT_QUERY_INTEGRAL)return QCORE_EXACT_YES;
        }
    }catch(...){return QCORE_EXACT_UNKNOWN;}
    return QCORE_EXACT_UNKNOWN;
}
inline int refine(std::uint32_t op,const qcore_exact_interval* args,std::uint32_t count,std::uint32_t bits,qcore_exact_interval* out) {
    if(!out||!spelling(op)||arity(op)!=count||(count&&!args)||bits>4096)return 0;
    try {
        Arithmetic a(std::max<std::uint32_t>(bits,16)+32);Interval value;
        if(op==1)value=a.sub(a.scale(a.atan_reciprocal(5),16),a.scale(a.atan_reciprocal(239),4));
        else if(op==2)value=a.exp(a.integer(1));
        else {
            auto x=a.import(args[0]);
            switch(op) {
                case 257: return 0; // Core canonicalizes square root to a rational power.
                case 258:value=a.trig(x,false);break;
                case 259:value=a.trig(x,true);break;
                case 260:value=a.div(a.trig(x,false),a.trig(x,true));break;
                case 261:value=a.log(x);break;
                case 262:value=a.exp(x);break;
                case 263:value=a.exp(a.mul(a.import(args[1]),a.log(x)));break;
                default:return 0;
            }
        }
        thread_local std::string low,high;low=value.low.decimal();high=value.high.decimal();
        out->low=low.c_str();out->high=high.c_str();out->exponent=-static_cast<std::int64_t>(a.precision);return 1;
    } catch (...) { return 0; }
}
inline const qcore_exact_provider_v2 provider{sizeof(qcore_exact_provider_v2),"math",spelling,arity,flags,canonicalize,prove,refine};

} // namespace math_exact
