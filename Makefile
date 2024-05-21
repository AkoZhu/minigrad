# specify the compiler
CXX = g++

# specify compiler flags
CXXFLAGS = -Wall -std=c++17 -Iinclude

# specify the target file
TARGET = train

# specify the source files
SRCS = train.cpp

# specify the object files
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<  -o $@

clean:
	$(RM) $(OBJS) $(TARGET)