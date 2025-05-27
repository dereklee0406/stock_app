"""
Main entry point for Stock Analysis Tool.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cli.cli import StockAppCLI

def main():
    cli = StockAppCLI()
    cli.run()

if __name__ == '__main__':
    main()
