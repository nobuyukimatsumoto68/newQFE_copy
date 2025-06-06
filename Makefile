
all: gen_grp

.phony: all clean gen_grp

clean:
	$(MAKE) -C grp clean

gen_grp:
	$(MAKE) -C grp
