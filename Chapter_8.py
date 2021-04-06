# There are at least two distinguishable kinds of errors:
# syntax errors and exceptions.

# Exceptions can be handled.
#while True:
#    try:
#        x = int(input("Please enter a number: "))
#        break
#    except ValueError:
#        print("Not a valid number. Try again.")

# Notes on the previous example: The try statement works as
# follows.
# 1) Execute the try clause, which is the group of statements
# between the try and except keywords.
# 2) If no exception occurs, the try clause is completed and
# the except clause is ignored.
# 3) If an exception occurs, the rest of the try clause is
# skipped. Then, if the exception type matches the exception
# named after the except keyword, the exception clause is
# executed.
# If there the type is not matched, it is passed on to outer
# try statements. If no handler is found, it is an unhandled
# exception and execution stops.

# Multiple exceptions may be named as a parenthesized tuple.
# except (RuntimeError, TypeError, NameError):
#     pass

# Except can be written without an exception name, to be used
# as a wildcard. This should be used with extreme caution
# since it can mask a real error. On the other hand, it can
# be used to print an error message then re-raise the
# exception.

# An else clause can be added to a try... except statement.
# This must follow all except clauses. This can be used for
# code that must be executed if the try clause does not raise
# an exception. This is better than adding additional code
# to the try clause because it avoids accidentally catching
# an exception that wasn't raised by the code being protected
# by the try... except statement.

# The exception itself can be defined as a variable and
# manipulated.
# except Exception as inst:

# New exceptions may be named by creating new exception
# classes.

# The try statement can also be followed by the additional
# optional clause of finally: which defines clean-up actions
# that must be executed under all circumstances.
def divide(x, y):
    try:
        result = x/y
    except ZeroDivisionError:
        print('A singularity!')
    else:
        print('The result is', result)
    finally:
        print('Thanks for playing!')

divide(2, 1)
divide(2, 0)
# divide('2', '1') This raises TypeError.

# Some objects offer predefined cleanup actions.
