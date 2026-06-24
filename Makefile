# VoxCPM2-C GNU Makefile (備援建置系統)
# 主要建置使用 CMake，此 Makefile 提供快速測試用
#
# 使用方式:
#   make          - 建置 Release (CPU)
#   make debug    - 建置 Debug
#   make test     - 建置並執行測試
#   make clean    - 清除建置檔案

CC       ?= gcc
CFLAGS   ?= -Wall -Wextra -Wpedantic -Wshadow -Wstrict-aliasing -std=c11 -Os
LDFLAGS  ?= -lm
AR       ?= ar

# 目錄
SRCDIR   := src
INCDIR   := include
BUILDDIR := build
TESTDIR  := tests

# 來源檔案
SRCS     := $(wildcard $(SRCDIR)/*.c)
OBJS     := $(patsubst $(SRCDIR)/%.c, $(BUILDDIR)/%.o, $(SRCS))
MAIN_OBJ := $(BUILDDIR)/main.o

# 測試來源
TEST_SRCS := $(wildcard $(TESTDIR)/*.c)
TEST_OBJS := $(filter-out $(MAIN_OBJ), $(OBJS)) \
             $(patsubst $(TESTDIR)/%.c, $(BUILDDIR)/%.o, $(TEST_SRCS))
TEST_BIN  := $(BUILDDIR)/test_runner

# 範例
EXAMPLES   := $(wildcard examples/*.c)
EXAMPLE_BINS := $(patsubst examples/%.c, $(BUILDDIR)/%, $(EXAMPLES))

.PHONY: all debug test clean mrproper

all: CFLAGS += -Os -DNDEBUG
all: $(BUILDDIR)/voxcpm2-c $(EXAMPLE_BINS)

debug: CFLAGS += -Og -g -DDEBUG -fsanitize=address
debug: LDFLAGS += -fsanitize=address
debug: $(BUILDDIR)/voxcpm2-c

test: CFLAGS += -Og -g -DDEBUG -fsanitize=address
test: LDFLAGS += -fsanitize=address
test: $(TEST_BIN)
	@echo "=== Running tests ==="
	$(TEST_BIN)

# 建置目錄
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# 物件檔規則
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/%.o: $(TESTDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -I$(INCDIR) -I$(TESTDIR) -c $< -o $@

# 主執行檔
$(BUILDDIR)/voxcpm2-c: $(OBJS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# 測試執行檔
$(TEST_BIN): $(TEST_OBJS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

# 範例
$(BUILDDIR)/%: examples/%.c $(filter-out $(MAIN_OBJ), $(OBJS))
	$(CC) $(CFLAGS) -I$(INCDIR) $< $(filter-out $(MAIN_OBJ), $(OBJS)) $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILDDIR)/*.o $(BUILDDIR)/voxcpm2-c $(BUILDDIR)/test_runner $(EXAMPLE_BINS)

mrproper: clean
	rm -rf $(BUILDDIR)
