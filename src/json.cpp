#include "footbatch/json.hpp"

#include <cmath>
#include <cstdio>

namespace footbatch {
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : s_(text), pos_(0) {}

    Json parse() {
        skipWs();
        Json value = parseValue();
        skipWs();
        if (pos_ != s_.size()) throw JsonError("trailing characters in json");
        return value;
    }

private:
    void skipWs() {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    char peek() {
        if (pos_ >= s_.size()) throw JsonError("unexpected end of json");
        return s_[pos_];
    }

    Json parseValue() {
        skipWs();
        char c = peek();
        if (c == '{') return parseObject();
        if (c == '[') return parseArray();
        if (c == '"') return Json::makeString(parseString());
        if (c == 't' || c == 'f') return parseBool();
        if (c == 'n') return parseNull();
        return parseNumber();
    }

    Json parseObject() {
        Json obj = Json::makeObject();
        ++pos_;
        skipWs();
        if (peek() == '}') { ++pos_; return obj; }
        while (true) {
            skipWs();
            if (peek() != '"') throw JsonError("expected string key in object");
            std::string key = parseString();
            skipWs();
            if (peek() != ':') throw JsonError("expected ':' after object key");
            ++pos_;
            obj.set(key, parseValue());
            skipWs();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; return obj; }
            throw JsonError("expected ',' or '}' in object");
        }
    }

    Json parseArray() {
        Json arr = Json::makeArray();
        ++pos_;
        skipWs();
        if (peek() == ']') { ++pos_; return arr; }
        while (true) {
            arr.push(parseValue());
            skipWs();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; return arr; }
            throw JsonError("expected ',' or ']' in array");
        }
    }

    std::string parseString() {
        ++pos_;
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= s_.size()) break;
                char e = s_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    default: throw JsonError("unsupported string escape");
                }
            } else {
                out.push_back(c);
            }
        }
        throw JsonError("unterminated string");
    }

    Json parseBool() {
        if (s_.compare(pos_, 4, "true") == 0) { pos_ += 4; return Json::makeBool(true); }
        if (s_.compare(pos_, 5, "false") == 0) { pos_ += 5; return Json::makeBool(false); }
        throw JsonError("invalid literal");
    }

    Json parseNull() {
        if (s_.compare(pos_, 4, "null") == 0) { pos_ += 4; return Json(); }
        throw JsonError("invalid literal");
    }

    Json parseNumber() {
        size_t start = pos_;
        if (peek() == '-') ++pos_;
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                ++pos_;
            } else {
                break;
            }
        }
        std::string token = s_.substr(start, pos_ - start);
        try {
            return Json::makeNumber(std::stod(token));
        } catch (...) {
            throw JsonError("invalid number: " + token);
        }
    }

    const std::string& s_;
    size_t pos_;
};

void writeString(std::ostringstream& os, const std::string& v) {
    os << '"';
    for (char c : v) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default: os << c; break;
        }
    }
    os << '"';
}

}  // namespace

Json Json::parse(const std::string& text) {
    return Parser(text).parse();
}

void Json::write(std::ostringstream& os, int indent, int depth) const {
    auto pad = [&](int d) {
        if (indent <= 0) return;
        for (int i = 0; i < indent * d; ++i) os << ' ';
    };
    switch (type_) {
        case Type::Null: os << "null"; break;
        case Type::Bool: os << (bool_ ? "true" : "false"); break;
        case Type::Number: {
            if (std::floor(num_) == num_ && std::abs(num_) < 1e15) {
                os << static_cast<long long>(num_);
            } else {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.10g", num_);
                os << buf;
            }
            break;
        }
        case Type::String: writeString(os, str_); break;
        case Type::Array: {
            if (arr_.empty()) { os << "[]"; break; }
            os << '[';
            for (size_t i = 0; i < arr_.size(); ++i) {
                if (i) os << ',';
                if (indent > 0) os << '\n';
                pad(depth + 1);
                arr_[i].write(os, indent, depth + 1);
            }
            if (indent > 0) { os << '\n'; pad(depth); }
            os << ']';
            break;
        }
        case Type::Object: {
            if (obj_.empty()) { os << "{}"; break; }
            os << '{';
            size_t i = 0;
            for (const auto& kv : obj_) {
                if (i++) os << ',';
                if (indent > 0) os << '\n';
                pad(depth + 1);
                writeString(os, kv.first);
                os << (indent > 0 ? ": " : ":");
                kv.second.write(os, indent, depth + 1);
            }
            if (indent > 0) { os << '\n'; pad(depth); }
            os << '}';
            break;
        }
    }
}

}  // namespace footbatch
