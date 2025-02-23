#include "sample.hpp"
#include <algorithm>
#include <sstream>

namespace data_processing {

template<typename T>
DataProcessor<T>::DataProcessor(size_t capacity)
    : capacity_(capacity)
    , count_(0) {
    items_.reserve(capacity);
}

template<typename T>
bool DataProcessor<T>::add_item(const T& item) {
    if (is_full()) {
        return false;
    }
    
    validate_item(item);
    items_.push_back(item);
    ++count_;
    return true;
}

template<typename T>
template<typename U>
std::vector<U> DataProcessor<T>::transform_data(const std::function<U(const T&)>& transformer) const {
    std::vector<U> result;
    result.reserve(count_);
    
    std::transform(items_.begin(), 
                  items_.begin() + count_,
                  std::back_inserter(result),
                  transformer);
    
    return result;
}

template<typename T>
void DataProcessor<T>::clear() noexcept {
    items_.clear();
    count_ = 0;
}

template<typename T>
bool DataProcessor<T>::is_full() const {
    return count_ >= capacity_;
}

template<typename T>
void DataProcessor<T>::validate_item(const T& item) {
    // Base implementation does no validation
}

StringProcessor::StringProcessor(size_t capacity)
    : DataProcessor<std::string>(capacity) {
}

void StringProcessor::process_batch(const std::vector<std::string>& batch) {
    for (const auto& item : batch) {
        if (!add_item(item)) {
            throw ProcessingError("Processor capacity exceeded during batch processing");
        }
    }
}

std::string StringProcessor::join(const std::string& delimiter) const {
    std::ostringstream result;
    auto transformed = transform_data<std::string>([](const std::string& s) { return s; });
    
    if (!transformed.empty()) {
        result << transformed[0];
        for (size_t i = 1; i < transformed.size(); ++i) {
            result << delimiter << transformed[i];
        }
    }
    
    return result.str();
}

void StringProcessor::validate_item(const std::string& item) {
    if (item.empty()) {
        throw ProcessingError("Empty strings are not allowed");
    }
    
    std::string normalized = item;
    normalize_string(normalized);
    if (normalized.empty()) {
        throw ProcessingError("String contains only whitespace");
    }
}

void StringProcessor::normalize_string(std::string& str) {
    // Remove leading/trailing whitespace
    str.erase(0, str.find_first_not_of(" \t\n\r"));
    str.erase(str.find_last_not_of(" \t\n\r") + 1);
}

ProcessingError::ProcessingError(const std::string& message)
    : std::runtime_error(message) {
}

ProcessingError ProcessingError::create_full_error(const std::string& context) {
    std::ostringstream error_msg;
    error_msg << "Processing error in context '" << context << "'";
    return ProcessingError(error_msg.str());
}

template<typename T>
std::unique_ptr<DataProcessor<T>> create_processor(size_t initial_capacity) {
    return std::make_unique<DataProcessor<T>>(initial_capacity);
}

// Explicit template instantiations
template class DataProcessor<std::string>;
template class DataProcessor<int>;
template std::unique_ptr<DataProcessor<std::string>> create_processor(size_t);
template std::unique_ptr<DataProcessor<int>> create_processor(size_t);
} 