#pragma once

#define DISALLOW_COPY_AND_ASSIGN(TypeName)      \
TypeName(const TypeName&) = delete;             \
void operator=(const TypeName&) = delete

#define DISALLOW_COPY_MOVE_AND_ASSIGN(TypeName) \
TypeName(const TypeName&) = delete;             \
void operator=(const TypeName&) = delete;       \
TypeName(TypeName&&) = delete;                  \
void operator=(TypeName&&) = delete

