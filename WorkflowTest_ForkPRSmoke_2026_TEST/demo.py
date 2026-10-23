import json
import sys


def main():
    numbers = [int(value) for value in sys.argv[1:]] or [1, 2, 3]
    print(json.dumps({"numbers": numbers, "sum": sum(numbers)}))


if __name__ == "__main__":
    main()
