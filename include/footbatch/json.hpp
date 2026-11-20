#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace footbatch {

class JsonError : public std::runtime_error {
public:
    explicit JsonError(const std::string& msg) : std::runtime_error(msg) {}
};

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() : type_(Type::Null), num_(0.0), bool_(false) {}
    static Json makeBool(bool v) { Json j; j.type_ = Type::Bool; j.bool_ = v; return j; }
    static Json makeNumber(double v) { Json j; j.type_ = Type::Number; j.num_ = v; return j; }
    static Json makeString(std::string v) { Json j; j.type_ = Type::String; j.str_ = std::move(v); return j; }
    static Json makeArray() { Json j; j.type_ = Type::Array; return j; }
    static Json makeObject() { Json j; j.type_ = Type::Object; return j; }

    Type type() const { return type_; }
    bool isNull() const { return type_ == Type::Null; }
    bool isObject() const { return type_ == Type::Object; }
    bool isArray() const { return type_ == Type::Array; }
    bool isNumber() const { return type_ == Type::Number; }
    bool isString() const { return type_ == Type::String; }

    double asNumber() const { require(Type::Number); return num_; }
    int asInt() const { return static_cast<int>(asNumber()); }
    const std::string& asString() const { require(Type::String); return str_; }
    bool asBool() const { require(Type::Bool); return bool_; }

    const std::vector<Json>& array() const { require(Type::Array); return arr_; }
    std::vector<Json>& array() { require(Type::Array); return arr_; }
    const std::map<std::string, Json>& object() const { require(Type::Object); return obj_; }

    bool contains(const std::string& key) const {
        return type_ == Type::Object && obj_.find(key) != obj_.end();
    }
    const Json& at(const std::string& key) const {
        require(Type::Object);
        auto it = obj_.find(key);
        if (it == obj_.end()) throw JsonError("missing key: " + key);
        return it->second;
    }
    Json& at(const std::string& key) {
        require(Type::Object);
        auto it = obj_.find(key);
        if (it == obj_.end()) throw JsonError("missing key: " + key);
        return it->second;
    }
    double numberOr(const std::string& key, double fallback) const {
        return contains(key) ? at(key).asNumber() : fallback;
    }
    std::string stringOr(const std::string& key, const std::string& fallback) const {
        return contains(key) ? at(key).asString() : fallback;
    }

    void push(Json v) { require(Type::Array); arr_.push_back(std::move(v)); }
    void set(const std::string& key, Json v) { require(Type::Object); obj_[key] = std::move(v); }

    std::string dump(int indent = 2) const {
        std::ostringstream os;
        write(os, indent, 0);
        os << "\n";
        return os.str();
    }

    static Json parse(const std::string& text);

private:
    void require(Type t) const {
        if (type_ != t) throw JsonError("json value has unexpected type");
    }
    void write(std::ostringstream& os, int indent, int depth) const;

    Type type_;
    double num_;
    bool bool_;
    std::string str_;
    std::vector<Json> arr_;
    std::map<std::string, Json> obj_;
};

}  // namespace footbatch
