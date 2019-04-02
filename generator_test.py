"""
生成器测试代码
"""

def test_generator():
    while True:
        for i in range(10):
            yield  i


if __name__ == '__main__':
    test_generator = test_generator()
    print(next(test_generator))
    print(next(test_generator))
    print(next(test_generator))
    print(next(test_generator))
    print(next(test_generator))
    print(next(test_generator))