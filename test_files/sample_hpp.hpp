#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace data_processing {

template<typename T>
class DataProcessor {
public:
    explicit DataProcessor(size_t capacity);
    virtual ~DataProcessor() = default;

    bool add_item(const T& item);
    
    template<typename U>
    std::vector<U> transform_data(const std::function<U(const T&)>& transformer) const;
    
    virtual void clear() noexcept;
    
    size_t get_count() const { return count_; }
    
    bool is_full() const;

protected:
    virtual void validate_item(const T& item);

private:
    std::vector<T> items_;
    size_t capacity_;
    size_t count_;
};

class StringProcessor : public DataProcessor<std::string> {
public:
    explicit StringProcessor(size_t capacity);
    
    void process_batch(const std::vector<std::string>& batch);
    
    std::string join(const std::string& delimiter = ", ") const;
    
protected:
    void validate_item(const std::string& item) override;
    
private:
    void normalize_string(std::string& str);
};

class ProcessingError : public std::runtime_error {
public:
    explicit ProcessingError(const std::string& message);
    
    static ProcessingError create_full_error(const std::string& context);
};

template<typename T>
std::unique_ptr<DataProcessor<T>> create_processor(size_t initial_capacity = 100);

} // namespace data_processing 