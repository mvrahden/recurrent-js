import { Assertable } from "./assertable";


describe('Assertable', () => {

  const sut = Assertable['assert'];

  it('no instance >> assert true statement >> should not throw an error', () => {
    const actual = () => { sut(true, 'Don\'t expect to throw.'); };
    
    expect(actual).not.toThrowError();
  });
  
  it('no instance >> assert true statement >> should throw an error', () => {
    const actual = () => { sut(false, 'Expect to throw.'); };

    expect(actual).toThrowError(/Expect to throw\./);
  });
});
