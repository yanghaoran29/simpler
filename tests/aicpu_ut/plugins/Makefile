# Build libinsn_count.so against a QEMU build tree with CONFIG_PLUGIN=y.
#
#   make
#   make QEMU_BUILD_DIR=/path/to/qemu-build
#
QEMU_BUILD_DIR ?= /data/y00955915/.local/src/qemu-6.2.0-build

include $(QEMU_BUILD_DIR)/config-host.mak

PLUGIN_CFLAGS := -fPIC -Wall $(GLIB_CFLAGS) -I$(SRC_PATH)/include/qemu
PLUGIN_CFLAGS += $(filter -W%, $(QEMU_CFLAGS))

all: libinsn_count.so

insn_count.o: insn_count.c
	$(CC) $(PLUGIN_CFLAGS) -c -o $@ $<

libinsn_count.so: insn_count.o
	$(CC) -shared -Wl,-soname,$@ -o $@ $^ $(GLIB_LIBS)

clean:
	rm -f insn_count.o libinsn_count.so

.PHONY: all clean
