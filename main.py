import argparse

import rdata

def main():
    parser = argparse.ArgumentParser(description="Lfmm demo")
    parser.add_argument("--data",
                        type=str,
                        required=True,
                        help="Path to the data"
    )
    args = parser.parse_args()
    
    # Import R data
    parsed = rdata.parser.parse_file(rdata.TESTDATA_PATH / args.data)
    converted = rdata.conversion.convert(parsed)
    example_data = converted['example.data']
    del converted
    
if __name__ == '__main__':
    main()